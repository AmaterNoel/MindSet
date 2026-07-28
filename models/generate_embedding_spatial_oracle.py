from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import (  # noqa: E402
    DEFAULT_ANNOTATION_DIR,
    DEFAULT_BETAS_1D_PATH,
    DEFAULT_CAPTION_EMBEDDINGS_PATH,
    DEFAULT_IMAGE_EMBEDDINGS_PATH,
    DEFAULT_SPLIT_SEED,
    DEFAULT_STIM_INFO_PATH,
    DEFAULT_STIMULUS_H5_PATH,
    create_dataloaders,
)
from models.generate_s7_g_controlnet import (  # noqa: E402
    create_pipe,
    evaluate_run,
    load_models,
    save_comparison,
    tensor_to_pil,
)
from models.train_base_model_1D import maybe_pool_fmri  # noqa: E402
from models.train_low_model_1D import batch_images_to_float  # noqa: E402
from models.train_low_model_structured import make_repeat_averaged_loaders  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate direct predicted image embeddings and oracle/predicted semantic + G spatial control."
    )
    parser.add_argument("--s7-checkpoint", type=Path, required=True)
    parser.add_argument("--g-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output/embedding_spatial_oracle_50"))
    parser.add_argument("--model-root", type=Path, default=ROOT / "save_pt")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--stim-info-path", type=Path, default=DEFAULT_STIM_INFO_PATH)
    parser.add_argument("--betas-1d-path", type=Path, default=DEFAULT_BETAS_1D_PATH)
    parser.add_argument("--caption-embeddings-path", type=Path, default=DEFAULT_CAPTION_EMBEDDINGS_PATH)
    parser.add_argument("--image-embeddings-path", type=Path, default=DEFAULT_IMAGE_EMBEDDINGS_PATH)
    parser.add_argument("--stimulus-h5-path", type=Path, default=DEFAULT_STIMULUS_H5_PATH)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--ip-adapter-scale", type=float, default=1.0)
    parser.add_argument("--controlnet-scale", type=float, default=0.8)
    parser.add_argument("--control-guidance-end", type=float, default=0.9)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def collect_records(args: argparse.Namespace, device: torch.device, semantic_model: Any, low_model: Any):
    loaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        annotation_dir=args.annotation_dir,
        stim_info_path=args.stim_info_path,
        betas_1d_path=args.betas_1d_path,
        caption_embeddings_path=args.caption_embeddings_path,
        image_embeddings_path=args.image_embeddings_path,
        stimulus_h5_path=args.stimulus_h5_path,
        fmri_format="1d",
        seed=args.seed,
        normalize="none",
        include_raw=True,
    )
    _, _, test_loader, _, _ = make_repeat_averaged_loaders(
        *loaders, batch_size=args.batch_size, num_workers=args.num_workers
    )
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in test_loader:
            fmri = batch["fmri"].to(device)
            semantic_fmri = maybe_pool_fmri(
                fmri,
                enable_pool=fmri.shape[-1] != semantic_model.config.in_dim,
                pool_num=semantic_model.config.in_dim,
                pool_type="max",
            )
            low_fmri = maybe_pool_fmri(
                fmri,
                enable_pool=fmri.shape[-1] != low_model.config.in_dim,
                pool_num=low_model.config.in_dim,
                pool_type="max",
            )
            predicted = semantic_model(semantic_fmri)["image_embedding"].float().cpu()
            low = low_model(low_fmri)["low_level_rgb"].float().cpu()
            target = batch["image_embeddings"].float().cpu()
            originals = batch_images_to_float(batch["image"], device).cpu()
            for index, metadata in enumerate(batch["metadata"]):
                records.append(
                    {
                        "predicted": predicted[index].clone(),
                        "oracle": target[index].clone(),
                        "low": low[index].clone(),
                        "original": originals[index].clone(),
                        "metadata": metadata,
                    }
                )
                if len(records) >= args.samples:
                    return records
    return records


def write_run(run_dir: Path, args: argparse.Namespace, method: str, manifest: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "method": method,
        "samples": len(manifest),
        "steps": args.steps,
        "ip_adapter_scale": args.ip_adapter_scale,
        "controlnet_scale": args.controlnet_scale,
        "control_guidance_end": args.control_guidance_end,
        "test_only": True,
        "caption_retrieval": False,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def generate(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    run_dir: Path,
    device: torch.device,
    embedding_key: str,
    use_spatial_control: bool,
) -> list[dict[str, Any]]:
    pipe = create_pipe(args, device, controlnet=use_spatial_control)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    manifest: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        name = f"{index:03d}_label{int(record['metadata']['label_index']):05d}_comparison.png"
        destination = run_dir / "gallery" / "test" / name
        if destination.exists():
            manifest.append({"split": "test", "file": f"gallery/test/{name}", **record["metadata"]})
            continue
        positive = record[embedding_key].reshape(1, 1, -1).to(device=device, dtype=dtype)
        embeds = torch.cat([torch.zeros_like(positive), positive], dim=0)
        kwargs: dict[str, Any] = {}
        if use_spatial_control:
            low = tensor_to_pil(record["low"]).resize(
                (args.image_size, args.image_size), Image.Resampling.BICUBIC
            )
            kwargs.update(
                image=low,
                controlnet_conditioning_scale=args.controlnet_scale,
                control_guidance_end=args.control_guidance_end,
            )
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        reconstruction = pipe(
            prompt="a natural photograph, realistic, sharp, coherent composition",
            negative_prompt="text, watermark, collage, duplicate, distorted, unfinished, blurry, low quality",
            ip_adapter_image_embeds=[embeds],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            height=args.image_size,
            width=args.image_size,
            **kwargs,
        ).images[0]
        subtitle = {
            ("predicted", False): "Predicted image embedding",
            ("oracle", True): "Oracle image embedding + G spatial",
            ("predicted", True): "Predicted image embedding + G spatial",
        }[(embedding_key, use_spatial_control)]
        save_comparison(tensor_to_pil(record["original"]), reconstruction, subtitle, destination)
        manifest.append({"split": "test", "file": f"gallery/test/{name}", **record["metadata"]})
    del pipe
    torch.cuda.empty_cache()
    return manifest


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    semantic_model, low_model = load_models(args, device)
    records = collect_records(args, device, semantic_model, low_model)
    del semantic_model, low_model
    torch.cuda.empty_cache()

    specs = [
        ("E1_predicted_embedding", "predicted", False, "Direct predicted image embedding via IP-Adapter"),
        ("E2_oracle_embedding_G_spatial", "oracle", True, "Oracle image embedding + G Tile spatial control"),
        ("E3_predicted_embedding_G_spatial", "predicted", True, "Predicted image embedding + G Tile spatial control"),
    ]
    run_dirs = []
    for name, key, spatial, method in specs:
        run_dir = args.output_root / name
        manifest = generate(args, records, run_dir, device, key, spatial)
        write_run(run_dir, args, method, manifest)
        run_dirs.append(run_dir)
    comparison = {run_dir.name: evaluate_run(args, run_dir, device) for run_dir in run_dirs}
    (args.output_root / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
