from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageDraw

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
from models.train_base_model_1D import Mind1D, Mind1DConfig, make_repeat_averaged_loaders  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from predicted fMRI semantics only.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=ROOT / "save_pt")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--stim-info-path", type=Path, default=DEFAULT_STIM_INFO_PATH)
    parser.add_argument("--betas-1d-path", type=Path, default=DEFAULT_BETAS_1D_PATH)
    parser.add_argument("--caption-embeddings-path", type=Path, default=DEFAULT_CAPTION_EMBEDDINGS_PATH)
    parser.add_argument("--image-embeddings-path", type=Path, default=DEFAULT_IMAGE_EMBEDDINGS_PATH)
    parser.add_argument("--stimulus-h5-path", type=Path, default=DEFAULT_STIMULUS_H5_PATH)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-split", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.9)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def select_fixed(loader: torch.utils.data.DataLoader, count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for batch in loader:
        for idx, metadata in enumerate(batch["metadata"]):
            label = int(metadata["label_index"])
            if label in seen:
                continue
            seen.add(label)
            selected.append(
                {
                    "fmri": batch["fmri"][idx].clone(),
                    "image": batch["image"][idx].clone(),
                    "target": batch["image_embeddings"][idx].clone(),
                    "metadata": metadata,
                }
            )
            if len(selected) == count:
                return selected
    return selected


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().numpy()
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def save_comparison(original: Image.Image, reconstruction: Image.Image, title: str, path: Path) -> None:
    size, label_h = 512, 48
    canvas = Image.new("RGB", (size * 2, size + label_h), "white")
    canvas.paste(original.resize((size, size), Image.Resampling.BICUBIC), (0, 0))
    canvas.paste(reconstruction.resize((size, size), Image.Resampling.BICUBIC), (size, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, size + 14), f"Original | {title}", fill="black")
    draw.text((size + 12, size + 14), "Semantic-only reconstruction", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

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
        include_vae_latents=False,
        include_raw=True,
    )
    train_loader, val_loader, test_loader, _, _ = make_repeat_averaged_loaders(
        *loaders, batch_size=args.batch_size, num_workers=args.num_workers
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = Mind1D(Mind1DConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    split_samples = {
        "train": select_fixed(train_loader, args.samples_per_split),
        "val": select_fixed(val_loader, args.samples_per_split),
        "test": select_fixed(test_loader, args.samples_per_split),
    }
    predicted: dict[str, list[torch.Tensor]] = {}
    prediction_metrics: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for split, samples in split_samples.items():
            fmri = torch.stack([item["fmri"] for item in samples]).to(device)
            output = model(fmri)["image_embedding"].float()
            target = F.normalize(torch.stack([item["target"] for item in samples]).to(device).float(), dim=-1)
            predicted[split] = list(output.cpu())
            cosine = F.cosine_similarity(output, target, dim=-1)
            logits = output @ target.T
            prediction_metrics[split] = {
                "clip_cosine_mean": float(cosine.mean().item()),
                "top1_within_fixed5": float((logits.argmax(dim=1) == torch.arange(len(samples), device=device)).float().mean().item()),
            }
    del model
    torch.cuda.empty_cache()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_root / "stabilityai__stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe.load_ip_adapter(
        args.model_root / "h94__IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
        image_encoder_folder="image_encoder",
        local_files_only=True,
    )
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    manifest = []
    for split, samples in split_samples.items():
        gallery_dir = args.run_dir / "gallery" / split
        for index, (sample, embedding) in enumerate(zip(samples, predicted[split])):
            positive = embedding.reshape(1, 1, 1280).to(device=device, dtype=dtype)
            embeds = torch.cat([torch.zeros_like(positive), positive], dim=0)
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            reconstruction = pipe(
                prompt="a natural photograph, realistic, coherent composition",
                negative_prompt="text, watermark, collage, duplicate, distorted, low quality",
                ip_adapter_image_embeds=[embeds],
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                height=args.image_size,
                width=args.image_size,
            ).images[0]
            metadata = sample["metadata"]
            name = f"{index:02d}_label{int(metadata['label_index']):05d}_comparison.png"
            save_comparison(
                tensor_to_pil(sample["image"]),
                reconstruction,
                f"label={metadata['label_index']} nsd={metadata['nsd_id']}",
                gallery_dir / name,
            )
            manifest.append({"split": split, "file": str((gallery_dir / name).relative_to(args.run_dir)), **metadata})

    with (args.run_dir / "generation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"prediction": prediction_metrics, "manifest": manifest}, handle, ensure_ascii=False, indent=2)
    print(json.dumps(prediction_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
