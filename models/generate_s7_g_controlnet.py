from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

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
from models.train_base_model_1D import Mind1D, Mind1DConfig, maybe_pool_fmri  # noqa: E402
from models.train_low_model_1D import batch_images_to_float  # noqa: E402
from models.train_low_model_structured import (  # noqa: E402
    StructuredConfig,
    StructuredLowModel,
    make_repeat_averaged_loaders,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and compare full-test S7, G and S7+G ControlNet results.")
    parser.add_argument("--s7-checkpoint", type=Path, required=True)
    parser.add_argument("--g-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output/fusion_model"))
    parser.add_argument("--model-root", type=Path, default=ROOT / "save_pt")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--stim-info-path", type=Path, default=DEFAULT_STIM_INFO_PATH)
    parser.add_argument("--betas-1d-path", type=Path, default=DEFAULT_BETAS_1D_PATH)
    parser.add_argument("--caption-embeddings-path", type=Path, default=DEFAULT_CAPTION_EMBEDDINGS_PATH)
    parser.add_argument("--image-embeddings-path", type=Path, default=DEFAULT_IMAGE_EMBEDDINGS_PATH)
    parser.add_argument("--stimulus-h5-path", type=Path, default=DEFAULT_STIMULUS_H5_PATH)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.9)
    parser.add_argument("--controlnet-scale", type=float, default=0.65)
    parser.add_argument("--control-guidance-end", type=float, default=0.8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def raw_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().numpy()
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def save_comparison(original: Image.Image, reconstruction: Image.Image, subtitle: str, path: Path) -> None:
    size, label_h = 512, 48
    canvas = Image.new("RGB", (size * 2, size + label_h), "white")
    canvas.paste(original.resize((size, size), Image.Resampling.BICUBIC), (0, 0))
    canvas.paste(reconstruction.resize((size, size), Image.Resampling.BICUBIC), (size, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, size + 14), "Original", fill="black")
    draw.text((size + 12, size + 14), subtitle, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def load_models(args: argparse.Namespace, device: torch.device) -> tuple[Mind1D, StructuredLowModel]:
    s7_payload = torch.load(args.s7_checkpoint, map_location="cpu")
    s7 = Mind1D(Mind1DConfig(**s7_payload["model_config"])).to(device)
    s7.load_state_dict(s7_payload["model_state_dict"])
    s7.eval()

    g_payload = torch.load(args.g_checkpoint, map_location="cpu")
    g = StructuredLowModel(StructuredConfig(**g_payload["model_config"])).to(device)
    g.load_state_dict(g_payload["model_state_dict"])
    g.eval()
    return s7, g


def collect_test_predictions(
    args: argparse.Namespace,
    device: torch.device,
    s7: Mind1D,
    g: StructuredLowModel,
) -> list[dict[str, Any]]:
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
            s7_fmri = maybe_pool_fmri(
                fmri,
                enable_pool=fmri.shape[-1] != s7.config.in_dim,
                pool_num=s7.config.in_dim,
                pool_type="max",
            )
            g_fmri = maybe_pool_fmri(
                fmri,
                enable_pool=fmri.shape[-1] != g.config.in_dim,
                pool_num=g.config.in_dim,
                pool_type="max",
            )
            semantic = s7(s7_fmri)["image_embedding"].float().cpu()
            low = g(g_fmri)["low_level_rgb"]
            assert isinstance(low, torch.Tensor)
            originals = batch_images_to_float(batch["image"], device).cpu()
            for index, metadata in enumerate(batch["metadata"]):
                records.append(
                    {
                        "semantic": semantic[index].clone(),
                        "low": low[index].float().cpu().clone(),
                        "original": originals[index].clone(),
                        "metadata": metadata,
                    }
                )
    return records


def create_pipe(args: argparse.Namespace, device: torch.device, controlnet: bool) -> Any:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        args.model_root / "madebyollin__sdxl-vae-fp16-fix",
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    if controlnet:
        control = ControlNetModel.from_pretrained(
            args.model_root / "xinsir__controlnet-tile-sdxl-1.0",
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=True,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            args.model_root / "stabilityai__stable-diffusion-xl-base-1.0",
            controlnet=control,
            vae=vae,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
            local_files_only=True,
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model_root / "stabilityai__stable-diffusion-xl-base-1.0",
            vae=vae,
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
    return pipe


def write_manifest(run_dir: Path, manifest: list[dict[str, Any]], config: dict[str, Any]) -> None:
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_g(records: list[dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    manifest = []
    for index, record in enumerate(records):
        metadata = record["metadata"]
        name = f"{index:03d}_label{int(metadata['label_index']):05d}_comparison.png"
        save_comparison(
            tensor_to_pil(record["original"]),
            tensor_to_pil(record["low"]),
            "G low-level reconstruction",
            run_dir / "gallery" / "test" / name,
        )
        manifest.append({"split": "test", "file": f"gallery/test/{name}", **metadata})
    return manifest


def generate_diffusion(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    run_dir: Path,
    device: torch.device,
    use_controlnet: bool,
) -> list[dict[str, Any]]:
    pipe = create_pipe(args, device, controlnet=use_controlnet)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    manifest = []
    for index, record in enumerate(records):
        positive = record["semantic"].reshape(1, 1, 1280).to(device=device, dtype=dtype)
        embeds = torch.cat([torch.zeros_like(positive), positive], dim=0)
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        kwargs: dict[str, Any] = {}
        if use_controlnet:
            kwargs.update(
                image=tensor_to_pil(record["low"]).resize(
                    (args.image_size, args.image_size), Image.Resampling.BICUBIC
                ),
                controlnet_conditioning_scale=args.controlnet_scale,
                control_guidance_end=args.control_guidance_end,
            )
        reconstruction = pipe(
            prompt="a natural photograph, realistic, coherent composition",
            negative_prompt="text, watermark, collage, duplicate, distorted, low quality",
            ip_adapter_image_embeds=[embeds],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            height=args.image_size,
            width=args.image_size,
            **kwargs,
        ).images[0]
        metadata = record["metadata"]
        name = f"{index:03d}_label{int(metadata['label_index']):05d}_comparison.png"
        subtitle = "S7 + G ControlNet" if use_controlnet else "S7 semantic reconstruction"
        save_comparison(
            tensor_to_pil(record["original"]),
            reconstruction,
            subtitle,
            run_dir / "gallery" / "test" / name,
        )
        manifest.append({"split": "test", "file": f"gallery/test/{name}", **metadata})
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return manifest


def image_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu_x = F.avg_pool2d(pred, 7, stride=1, padding=3)
    mu_y = F.avg_pool2d(target, 7, stride=1, padding=3)
    sigma_x = F.avg_pool2d(pred * pred, 7, stride=1, padding=3) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, 7, stride=1, padding=3) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred * target, 7, stride=1, padding=3) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-6)
    return score.flatten(1).mean(1)


def evaluate_run(args: argparse.Namespace, run_dir: Path, device: torch.device) -> dict[str, float]:
    files = sorted((run_dir / "gallery" / "test").glob("*_comparison.png"))
    originals, reconstructions = [], []
    for path in files:
        image = Image.open(path).convert("RGB")
        originals.append(image.crop((0, 0, 512, 512)))
        reconstructions.append(image.crop((512, 0, 1024, 512)))

    pixel_ssim, pixel_psnr = [], []
    for start in range(0, len(files), 16):
        original = torch.stack(
            [torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1) / 255 for im in originals[start : start + 16]]
        ).to(device)
        recon = torch.stack(
            [torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1) / 255 for im in reconstructions[start : start + 16]]
        ).to(device)
        pixel_ssim.extend(image_ssim(recon, original).cpu().tolist())
        mse = (recon - original).square().flatten(1).mean(1).clamp_min(1e-10)
        pixel_psnr.extend((-10.0 * torch.log10(mse)).cpu().tolist())

    clip = CLIPModel.from_pretrained(
        args.model_root / "openai__clip-vit-base-patch32", local_files_only=True
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        args.model_root / "openai__clip-vit-base-patch32", local_files_only=True
    )
    clip.eval()
    cosines = []
    with torch.no_grad():
        for start in range(0, len(files), 32):
            batch_images = originals[start : start + 32] + reconstructions[start : start + 32]
            pixels = processor(images=batch_images, return_tensors="pt")["pixel_values"].to(device)
            output = clip.vision_model(pixel_values=pixels)
            features = F.normalize(clip.visual_projection(output.pooler_output).float(), dim=-1)
            count = len(batch_images) // 2
            cosines.extend(F.cosine_similarity(features[:count], features[count:], dim=-1).cpu().tolist())
    del clip
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics = {
        "samples": len(files),
        "ssim_mean": float(np.mean(pixel_ssim)),
        "ssim_std": float(np.std(pixel_ssim)),
        "psnr_mean": float(np.mean(pixel_psnr)),
        "psnr_std": float(np.std(pixel_psnr)),
        "clip_b32_cosine_mean": float(np.mean(cosines)),
        "clip_b32_cosine_std": float(np.std(cosines)),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    s7, g = load_models(args, device)
    records = collect_test_predictions(args, device, s7, g)
    del s7, g
    if device.type == "cuda":
        torch.cuda.empty_cache()

    common = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "test_samples": len(records),
        "cosine_definition": "CLIP ViT-B/32 image embedding cosine between reconstruction and original",
    }
    runs = {
        "G": args.output_root / "G_full_test",
        "S7": args.output_root / "S7_full_test",
        "S7+G": args.output_root / "S7_G_controlnet_full_test",
    }
    g_manifest = generate_g(records, runs["G"])
    write_manifest(runs["G"], g_manifest, {**common, "method": "G low-level RGB"})
    s7_manifest = generate_diffusion(args, records, runs["S7"], device, use_controlnet=False)
    write_manifest(runs["S7"], s7_manifest, {**common, "method": "S7 semantic IP-Adapter"})
    joint_manifest = generate_diffusion(args, records, runs["S7+G"], device, use_controlnet=True)
    write_manifest(
        runs["S7+G"],
        joint_manifest,
        {**common, "method": "S7 semantic IP-Adapter + G tile ControlNet", "controlnet_scale": args.controlnet_scale},
    )

    summary = {name: evaluate_run(args, run_dir, device) for name, run_dir in runs.items()}
    (args.output_root / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
