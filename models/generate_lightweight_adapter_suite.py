from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import (  # noqa: E402
    DEFAULT_ANNOTATION_DIR,
    DEFAULT_BETAS_1D_PATH,
    DEFAULT_CAPTION_EMBEDDINGS_PATH,
    DEFAULT_IMAGE_EMBEDDINGS_PATH,
    DEFAULT_STIM_INFO_PATH,
    DEFAULT_STIMULUS_H5_PATH,
)
from models.generate_embedding_spatial_oracle import collect_records, write_run  # noqa: E402
from models.generate_s7_g_controlnet import (  # noqa: E402
    create_pipe,
    load_models,
    save_comparison,
    tensor_to_pil,
)
from models.train_lightweight_fusion_adapters import (  # noqa: E402
    SemanticResidualAdapter,
    SpatialRGBAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s7-checkpoint", type=Path, required=True)
    parser.add_argument("--g-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def generate(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    run_dir: Path,
    device: torch.device,
    embedding_key: str,
    control_key: str | None,
    subtitle: str,
) -> list[dict[str, Any]]:
    pipe = create_pipe(args, device, controlnet=control_key is not None)
    dtype = torch.float16
    manifest = []
    for index, record in enumerate(records):
        name = f"{index:03d}_label{int(record['metadata']['label_index']):05d}_comparison.png"
        destination = run_dir / "gallery" / "test" / name
        if not destination.exists():
            positive = record[embedding_key].reshape(1, 1, -1).to(device=device, dtype=dtype)
            embeds = torch.cat([torch.zeros_like(positive), positive], dim=0)
            kwargs: dict[str, Any] = {}
            if control_key is not None:
                kwargs.update(
                    image=tensor_to_pil(record[control_key]).resize(
                        (args.image_size, args.image_size), Image.Resampling.BICUBIC
                    ),
                    controlnet_conditioning_scale=args.controlnet_scale,
                    control_guidance_end=args.control_guidance_end,
                )
            image = pipe(
                prompt="a natural photograph, realistic, sharp, coherent composition",
                negative_prompt="text, watermark, collage, duplicate, distorted, unfinished, blurry, low quality",
                ip_adapter_image_embeds=[embeds],
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device=device).manual_seed(args.seed + index),
                height=args.image_size,
                width=args.image_size,
                **kwargs,
            ).images[0]
            save_comparison(tensor_to_pil(record["original"]), image, subtitle, destination)
        manifest.append({"split": "test", "file": f"gallery/test/{name}", **record["metadata"]})
    del pipe
    torch.cuda.empty_cache()
    return manifest


def extract_features(
    images: list[Image.Image], model: CLIPModel, processor: CLIPProcessor, device: torch.device
) -> torch.Tensor:
    features = []
    with torch.no_grad():
        for start in range(0, len(images), 24):
            pixels = processor(images=images[start : start + 24], return_tensors="pt")[
                "pixel_values"
            ].to(device)
            output = model.vision_model(pixel_values=pixels)
            features.append(model.visual_projection(output.pooler_output).float().cpu())
    return F.normalize(torch.cat(features), dim=-1)


def evaluate_public(args: argparse.Namespace, run_dir: Path, device: torch.device) -> dict[str, float]:
    files = sorted((run_dir / "gallery" / "test").glob("*_comparison.png"))
    originals, recons = [], []
    for path in files:
        paired = Image.open(path).convert("RGB")
        originals.append(paired.crop((0, 0, 512, 512)))
        recons.append(paired.crop((512, 0, 1024, 512)))
    pixcorr, ssim, psnr = [], [], []
    for original, recon in zip(originals, recons, strict=True):
        x = np.asarray(original.resize((425, 425), Image.Resampling.BICUBIC), dtype=np.float32) / 255
        y = np.asarray(recon.resize((425, 425), Image.Resampling.BICUBIC), dtype=np.float32) / 255
        pixcorr.append(float(np.corrcoef(x.reshape(-1), y.reshape(-1))[0, 1]))
        ssim.append(float(structural_similarity(x, y, channel_axis=2, data_range=1.0)))
        mse = max(float(np.mean((x - y) ** 2)), 1e-10)
        psnr.append(float(-10 * np.log10(mse)))

    b32 = CLIPModel.from_pretrained(
        args.model_root / "openai__clip-vit-base-patch32", local_files_only=True
    ).to(device)
    b32_proc = CLIPProcessor.from_pretrained(
        args.model_root / "openai__clip-vit-base-patch32", local_files_only=True
    )
    original_b32 = extract_features(originals, b32, b32_proc, device)
    recon_b32 = extract_features(recons, b32, b32_proc, device)
    cosine = F.cosine_similarity(original_b32, recon_b32, dim=-1).numpy()
    del b32
    torch.cuda.empty_cache()

    vitl = CLIPModel.from_pretrained(
        args.model_root / "openai__clip-vit-large-patch14", local_files_only=True
    ).to(device)
    vitl_proc = CLIPProcessor.from_pretrained(
        args.model_root / "openai__clip-vit-large-patch14", local_files_only=True
    )
    original_l = extract_features(originals, vitl, vitl_proc, device)
    recon_l = extract_features(recons, vitl, vitl_proc, device)
    similarity = recon_l @ original_l.T
    diagonal = similarity.diag()[:, None]
    mask = ~torch.eye(len(files), dtype=torch.bool)
    two_way = (diagonal > similarity)[mask].float().mean().item()
    del vitl
    torch.cuda.empty_cache()

    result = {
        "samples": len(files),
        "pixcorr_mean": float(np.mean(pixcorr)),
        "pixcorr_std": float(np.std(pixcorr)),
        "ssim_425_skimage_mean": float(np.mean(ssim)),
        "ssim_425_skimage_std": float(np.std(ssim)),
        "psnr_425_mean": float(np.mean(psnr)),
        "clip_b32_cosine_mean": float(np.mean(cosine)),
        "clip_b32_cosine_std": float(np.std(cosine)),
        "clip_vitl14_two_way_identification": float(two_way),
        "protocol": "MindEye public protocol: 425px PixCorr/SSIM; OpenAI CLIP ViT-L/14 two-way identification",
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    s7, g = load_models(args, device)
    records = collect_records(args, device, s7, g)
    payload = torch.load(args.adapter_checkpoint, map_location="cpu")
    semantic = SemanticResidualAdapter().to(device)
    spatial = SpatialRGBAdapter().to(device)
    semantic.load_state_dict(payload["semantic_adapter"])
    spatial.load_state_dict(payload["spatial_adapter"])
    semantic.eval()
    spatial.eval()
    with torch.no_grad():
        for record in records:
            record["adapted_semantic"] = semantic(record["predicted"].to(device)[None])[0].cpu()
            record["adapted_low"] = spatial(record["low"].to(device)[None])[0].cpu()
    del s7, g, semantic, spatial
    torch.cuda.empty_cache()

    specs = [
        ("N1_S7_semantic_adapter", "adapted_semantic", None, "S7 + semantic adapter"),
        ("N2_oracle_G_spatial_adapter", "oracle", "adapted_low", "Oracle + G spatial adapter"),
        ("N3_S7_G_dual_adapter", "adapted_semantic", "adapted_low", "S7 + G dual adapter"),
    ]
    run_dirs = []
    for name, embedding, control, subtitle in specs:
        run_dir = args.output_root / name
        manifest = generate(args, records, run_dir, device, embedding, control, subtitle)
        write_run(run_dir, args, subtitle, manifest)
        run_dirs.append(run_dir)
    comparison = {run.name: evaluate_public(args, run, device) for run in run_dirs}
    (args.output_root / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
