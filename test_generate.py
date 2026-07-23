import argparse
import csv
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import open_clip
import torch
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = ROOT / "save_pt"
DEFAULT_NSD_ROOT = Path(r"D:\datasets\NSD")
DEFAULT_STIMULI_H5 = DEFAULT_NSD_ROOT / "stimulus" / "S1_stimuli_256.h5py"
DEFAULT_ANNOTATION_DIR = DEFAULT_NSD_ROOT / "annotations"
DEFAULT_STIM_INFO = DEFAULT_ANNOTATION_DIR / "nsd_stim_info_merged.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate with SDXL + ControlNet Tile using five caption embeddings as IP-Adapter semantics."
    )
    parser.add_argument("-i", "--index", type=int, default=1, help="Stimulus index in S1_stimuli_256.h5py.")
    parser.add_argument("--stimuli-h5", type=Path, default=DEFAULT_STIMULI_H5)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--stim-info", type=Path, default=DEFAULT_STIM_INFO)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--blur-size", type=int, default=16, help="Downsample size before upsampling to image-size.")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--controlnet-scale", type=float, default=0.1)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="a natural photograph, high detail")
    parser.add_argument("--negative-prompt", type=str, default="low quality, blurry, distorted, text, watermark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Load models and caption embeddings, then skip sampling.")
    return parser.parse_args()


def load_stimulus(path: Path, index: int) -> Image.Image:
    with h5py.File(path, "r") as f:
        data = f["stimuli"]
        if index < 0 or index >= data.shape[0]:
            raise IndexError(f"index {index} is outside [0, {data.shape[0] - 1}]")
        image = data[index]

    if image.ndim != 3:
        raise ValueError(f"Expected CHW or HWC image, got shape {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = np.asarray(image, dtype=np.uint8)
    return Image.fromarray(image, mode="RGB")


def make_blurry_anchor(image: Image.Image, size: int, blur_size: int) -> Image.Image:
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    low = image.resize((blur_size, blur_size), Image.Resampling.BILINEAR)
    return low.resize((size, size), Image.Resampling.BICUBIC)


def load_captions_by_image(annotation_dir: Path) -> dict[str, dict[int, list[str]]]:
    captions_by_split: dict[str, dict[int, list[str]]] = {}
    for split in ("train2017", "val2017"):
        path = annotation_dir / f"captions_{split}.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        captions_by_image: dict[int, list[str]] = defaultdict(list)
        for ann in data.get("annotations", []):
            image_id = int(ann["image_id"])
            caption = str(ann["caption"]).strip()
            if caption and len(captions_by_image[image_id]) < 5:
                captions_by_image[image_id].append(caption)
        captions_by_split[split] = dict(captions_by_image)
    return captions_by_split


def load_stimulus_record(
    stim_info: Path,
    annotation_dir: Path,
    subject: int,
    index: int,
) -> tuple[dict[str, str], list[str]]:
    subject_col = f"subject{subject}"
    with stim_info.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    records = [
        row
        for split in ("train2017", "val2017")
        for row in all_rows
        if row["cocoSplit"] == split and int(row[subject_col]) == 1
    ]

    if index < 0 or index >= len(records):
        raise IndexError(f"index {index} is outside [0, {len(records) - 1}] for subject {subject}")

    record = records[index]
    captions_by_split = load_captions_by_image(annotation_dir)
    coco_split = record["cocoSplit"]
    coco_id = int(record["cocoId"])
    captions = captions_by_split[coco_split].get(coco_id, [])
    if len(captions) < 5:
        raise RuntimeError(f"Expected 5 captions for {coco_split}/{coco_id}, got {len(captions)}.")
    return record, captions[:5]


def load_openclip_text_model(args: argparse.Namespace, dtype: torch.dtype):
    model_dir = args.model_root / "laion__CLIP-ViT-bigG-14-laion2B-39B-b160k"
    weights = model_dir / "open_clip_model.safetensors"
    precision = "fp16" if dtype == torch.float16 else "fp32"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14",
        pretrained=str(weights),
        device=args.device,
        precision=precision,
    )
    tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
    model.eval()
    return model, tokenizer


@torch.no_grad()
def encode_caption_embeddings(args: argparse.Namespace, captions: list[str], dtype: torch.dtype) -> torch.Tensor:
    model, tokenizer = load_openclip_text_model(args, dtype)
    tokens = tokenizer(captions).to(args.device)
    embeddings = model.encode_text(tokens)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    embeddings = embeddings.to(dtype=dtype).detach().cpu()
    del model, tokenizer, tokens
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return embeddings


def load_pipeline(args: argparse.Namespace, dtype: torch.dtype) -> StableDiffusionXLControlNetImg2ImgPipeline:
    base_dir = args.model_root / "stabilityai__stable-diffusion-xl-base-1.0"
    controlnet_dir = args.model_root / "xinsir__controlnet-tile-sdxl-1.0"
    vae_dir = args.model_root / "madebyollin__sdxl-vae-fp16-fix"
    ip_adapter_dir = args.model_root / "h94__IP-Adapter"

    controlnet = ControlNetModel.from_pretrained(
        controlnet_dir,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(
        vae_dir,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_dir,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe.load_ip_adapter(
        ip_adapter_dir,
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
        image_encoder_folder="image_encoder",
        local_files_only=True,
    )
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)
    pipe.set_progress_bar_config(disable=False)

    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.cpu_offload and args.device.startswith("cuda"):
        pipe.enable_model_cpu_offload(device=args.device)
    else:
        pipe.to(args.device)
    pipe.vae.enable_tiling()
    return pipe


def make_ip_adapter_embeds(
    embedding: torch.Tensor,
    dtype: torch.dtype,
    do_classifier_free_guidance: bool,
) -> list[torch.Tensor]:
    positive = embedding.reshape(1, 1280).to(dtype=dtype)
    if do_classifier_free_guidance:
        negative = torch.zeros_like(positive)
        embeds = torch.cat([negative, positive], dim=0).unsqueeze(1)
    else:
        embeds = positive.unsqueeze(1)
    return [embeds]


def wrap_text(text: str, width: int = 34, max_lines: int = 4) -> list[str]:
    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return lines


def save_result_grid(
    original: Image.Image,
    anchor: Image.Image,
    reconstructions: list[Image.Image],
    captions: list[str],
    output: Path,
) -> None:
    panel_size = 256
    label_h = 104
    gap = 12
    columns = [("Original", original), ("Blurry anchor", anchor)]
    columns.extend((caption, image) for caption, image in zip(captions, reconstructions))

    width = len(columns) * panel_size + (len(columns) - 1) * gap
    height = panel_size + label_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    x = 0
    for label, image in columns:
        image = image.resize((panel_size, panel_size), Image.Resampling.BICUBIC)
        canvas.paste(image, (x, 0))
        y = panel_size + 8
        for line in wrap_text(label):
            draw.text((x + 6, y), line, fill=(0, 0, 0))
            y += 16
        x += panel_size + gap

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    original = load_stimulus(args.stimuli_h5, args.index)
    anchor = make_blurry_anchor(original, args.image_size, args.blur_size)
    record, captions = load_stimulus_record(args.stim_info, args.annotation_dir, args.subject, args.index)

    print(f"stimulus_index: {args.index}")
    print(f"nsd_id: {record['nsdId']}")
    print(f"coco_id: {record['cocoId']}")
    print(f"coco_split: {record['cocoSplit']}")
    print(f"original: {original.size}")
    print(f"anchor_blurry: {anchor.size}, blur_size={args.blur_size}")
    for idx, caption in enumerate(captions, start=1):
        print(f"caption_{idx}: {caption}")

    caption_embeddings = encode_caption_embeddings(args, captions, dtype)
    print(f"caption_embeddings: {tuple(caption_embeddings.shape)}")

    pipe = load_pipeline(args, dtype)
    if args.dry_run:
        print("dry_run: models and caption embeddings loaded; sampling skipped.")
        return

    reconstructions: list[Image.Image] = []
    for idx, embedding in enumerate(caption_embeddings):
        ip_adapter_embeds = make_ip_adapter_embeds(
            embedding,
            dtype=dtype,
            do_classifier_free_guidance=args.guidance_scale > 1.0,
        )
        generator = torch.Generator(device=args.device).manual_seed(args.seed + idx)
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=anchor,
            control_image=anchor,
            ip_adapter_image_embeds=ip_adapter_embeds,
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            generator=generator,
            height=args.image_size,
            width=args.image_size,
        ).images[0]
        reconstructions.append(result)

    output = args.output
    if output is None:
        output = ROOT / "output" / f"caption_tile_test_{args.index:05d}.png"
    save_result_grid(original, anchor, reconstructions, captions, output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
