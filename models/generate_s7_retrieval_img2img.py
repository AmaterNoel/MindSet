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
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import create_dataloaders  # noqa: E402
from models.generate_s7_g_controlnet import (  # noqa: E402
    collect_test_predictions,
    evaluate_run,
    load_models,
    save_comparison,
    tensor_to_pil,
)
from models.train_base_model_1D import make_repeat_averaged_loaders  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S7 caption retrieval and G-anchored SDXL experiments.")
    parser.add_argument("--s7-checkpoint", type=Path, required=True)
    parser.add_argument("--g-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output/fusion_retrieval_50"))
    parser.add_argument("--model-root", type=Path, default=ROOT / "save_pt")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--controlnet-scale", type=float, default=0.25)
    parser.add_argument("--control-guidance-end", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_vae(args: argparse.Namespace, dtype: torch.dtype) -> AutoencoderKL:
    return AutoencoderKL.from_pretrained(
        args.model_root / "madebyollin__sdxl-vae-fp16-fix",
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )


def make_text_pipe(args: argparse.Namespace, device: torch.device) -> StableDiffusionXLPipeline:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_root / "stabilityai__stable-diffusion-xl-base-1.0",
        vae=make_vae(args, dtype),
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def make_img2img_pipe(args: argparse.Namespace, device: torch.device) -> StableDiffusionXLImg2ImgPipeline:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.model_root / "stabilityai__stable-diffusion-xl-base-1.0",
        vae=make_vae(args, dtype),
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def make_control_pipe(
    args: argparse.Namespace,
    device: torch.device,
) -> StableDiffusionXLControlNetImg2ImgPipeline:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        args.model_root / "xinsir__controlnet-tile-sdxl-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        args.model_root / "stabilityai__stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=make_vae(args, dtype),
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def caption_bank(args: argparse.Namespace) -> tuple[torch.Tensor, list[str], list[int]]:
    train_loader, _, _ = create_dataloaders(
        batch_size=256,
        num_workers=0,
        fmri_format="1d",
        seed=args.seed,
        normalize="none",
        include_raw=False,
        include_vae_latents=False,
    )
    dataset = train_loader.dataset
    embeddings: list[torch.Tensor] = []
    captions: list[str] = []
    labels: list[int] = []
    seen: set[int] = set()
    for record_index in dataset.indices:
        record = dataset.records[record_index]
        label = int(record["label_index"])
        if label in seen:
            continue
        seen.add(label)
        for caption_index, valid in enumerate(dataset.caption_mask[label].tolist()):
            text = str(record["captions"][caption_index]).strip()
            if valid and text:
                embeddings.append(dataset.caption_text_embeddings[label, caption_index].float())
                captions.append(text)
                labels.append(label)
    return F.normalize(torch.stack(embeddings), dim=-1), captions, labels


def retrieve_prompts(
    records: list[dict[str, Any]],
    bank: torch.Tensor,
    captions: list[str],
    labels: list[int],
    top_k: int,
) -> list[dict[str, Any]]:
    predictions = F.normalize(torch.stack([record["semantic"] for record in records]).float(), dim=-1)
    scores = predictions @ bank.T
    results = []
    for row in range(scores.shape[0]):
        values, indices = scores[row].topk(min(max(top_k * 4, top_k), scores.shape[1]))
        selected: list[dict[str, Any]] = []
        selected_text: set[str] = set()
        selected_labels: set[int] = set()
        for score, index in zip(values.tolist(), indices.tolist(), strict=True):
            text = captions[index]
            label = labels[index]
            normalized = text.lower().strip()
            if normalized in selected_text or label in selected_labels:
                continue
            selected.append({"caption": text, "score": score, "train_label": label})
            selected_text.add(normalized)
            selected_labels.add(label)
            if len(selected) == top_k:
                break
        prompt = ". ".join(item["caption"].rstrip(".") for item in selected)
        results.append({"prompt": prompt, "retrieved": selected})
    return results


def save_result(
    run_dir: Path,
    index: int,
    record: dict[str, Any],
    reconstruction: Any,
    subtitle: str,
) -> dict[str, Any]:
    metadata = record["metadata"]
    name = f"{index:03d}_label{int(metadata['label_index']):05d}_comparison.png"
    save_comparison(
        tensor_to_pil(record["original"]),
        reconstruction,
        subtitle,
        run_dir / "gallery" / "test" / name,
    )
    return {"split": "test", "file": f"gallery/test/{name}", **metadata}


def generate_text(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    retrievals: list[dict[str, Any]],
    run_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    pipe = make_text_pipe(args, device)
    manifest = []
    for index, (record, retrieval) in enumerate(zip(records, retrievals, strict=True)):
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        image = pipe(
            prompt=retrieval["prompt"],
            negative_prompt="text, watermark, collage, duplicate, distorted, low quality",
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            height=args.image_size,
            width=args.image_size,
        ).images[0]
        manifest.append(save_result(run_dir, index, record, image, "S7 retrieved-caption SDXL"))
    del pipe
    torch.cuda.empty_cache()
    return manifest


def generate_img2img(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    retrievals: list[dict[str, Any]],
    strengths: list[float],
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    pipe = make_img2img_pipe(args, device)
    manifests: dict[str, list[dict[str, Any]]] = {}
    for strength in strengths:
        key = f"img2img_s{int(round(strength * 100)):03d}"
        run_dir = args.output_root / key
        manifest = []
        for index, (record, retrieval) in enumerate(zip(records, retrievals, strict=True)):
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            init_image = tensor_to_pil(record["low"]).resize(
                (args.image_size, args.image_size), resample=3
            )
            image = pipe(
                prompt=retrieval["prompt"],
                negative_prompt="text, watermark, collage, duplicate, distorted, low quality",
                image=init_image,
                strength=strength,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            manifest.append(save_result(run_dir, index, record, image, f"G img2img strength={strength:.2f}"))
        manifests[key] = manifest
    del pipe
    torch.cuda.empty_cache()
    return manifests


def generate_weak_control(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    retrievals: list[dict[str, Any]],
    run_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    pipe = make_control_pipe(args, device)
    manifest = []
    for index, (record, retrieval) in enumerate(zip(records, retrievals, strict=True)):
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        init_image = tensor_to_pil(record["low"]).resize((args.image_size, args.image_size), resample=3)
        image = pipe(
            prompt=retrieval["prompt"],
            negative_prompt="text, watermark, collage, duplicate, distorted, low quality",
            image=init_image,
            control_image=init_image,
            strength=0.4,
            controlnet_conditioning_scale=args.controlnet_scale,
            control_guidance_end=args.control_guidance_end,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        manifest.append(save_result(run_dir, index, record, image, "G img2img + weak Tile ControlNet"))
    del pipe
    torch.cuda.empty_cache()
    return manifest


def write_run(
    args: argparse.Namespace,
    run_dir: Path,
    method: str,
    manifest: list[dict[str, Any]],
    retrievals: list[dict[str, Any]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "method": method,
                "samples": args.samples,
                "top_k": args.top_k,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "controlnet_scale": args.controlnet_scale,
                "control_guidance_end": args.control_guidance_end,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "retrievals.json").write_text(
        json.dumps(retrievals, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    s7, g = load_models(args, device)
    records = collect_test_predictions(args, device, s7, g)[: args.samples]
    del s7, g
    torch.cuda.empty_cache()
    bank, captions, labels = caption_bank(args)
    retrievals = retrieve_prompts(records, bank, captions, labels, args.top_k)

    text_dir = args.output_root / "S7_retrieval_text"
    text_manifest = generate_text(args, records, retrievals, text_dir, device)
    write_run(args, text_dir, "S7 Top-K train-caption retrieval + SDXL text", text_manifest, retrievals)

    img_manifests = generate_img2img(args, records, retrievals, [0.25, 0.40, 0.55], device)
    for key, manifest in img_manifests.items():
        write_run(args, args.output_root / key, f"G-initialized SDXL img2img {key}", manifest, retrievals)

    control_dir = args.output_root / "img2img_s040_weak_control"
    control_manifest = generate_weak_control(args, records, retrievals, control_dir, device)
    write_run(
        args,
        control_dir,
        "G img2img strength=0.40 + weak Tile ControlNet",
        control_manifest,
        retrievals,
    )

    run_dirs = [text_dir, *(args.output_root / key for key in img_manifests), control_dir]
    comparison = {run_dir.name: evaluate_run(args, run_dir, device) for run_dir in run_dirs}
    (args.output_root / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
