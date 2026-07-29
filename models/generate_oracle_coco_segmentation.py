from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import DEFAULT_ANNOTATION_DIR, DEFAULT_SPLIT_SEED, create_dataloaders
from models.train_low_model_structured import make_repeat_averaged_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle COCO segmentation ControlNet test.")
    parser.add_argument("--output-root", type=Path, default=Path("output/oracle_coco_seg_5"))
    parser.add_argument("--model-root", type=Path, default=ROOT / "save_pt")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--controlnet-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def unique_test_items(samples: int) -> list[dict[str, Any]]:
    loaders = create_dataloaders(
        batch_size=1,
        num_workers=0,
        fmri_format="1d",
        normalize="none",
        include_raw=True,
        include_vae_latents=False,
    )
    _, _, test_loader, _, _ = make_repeat_averaged_loaders(
        *loaders, batch_size=1, num_workers=0
    )
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for batch in test_loader:
        metadata = batch["metadata"][0]
        nsd_id = int(metadata["nsd_id"])
        if nsd_id in seen:
            continue
        seen.add(nsd_id)
        selected.append(
            {
                "metadata": metadata,
                "image": batch["image"][0],
                "captions": list(batch["captions"][0]),
                "caption_mask": batch["caption_mask"][0].bool().tolist(),
            }
        )
        if len(selected) == samples:
            break
    if len(selected) != samples:
        raise RuntimeError(f"Only found {len(selected)} unique test samples.")
    return selected


def make_segment_condition(
    coco: COCO,
    image_id: int,
    size: tuple[int, int],
    category_colors: dict[int, list[int]],
) -> tuple[Image.Image, list[dict[str, Any]]]:
    width, height = size
    condition = np.zeros((height, width, 3), dtype=np.uint8)
    records: list[dict[str, Any]] = []
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=None))
    # Larger regions first and smaller instances last so small objects remain visible.
    annotations.sort(key=lambda ann: float(ann.get("area", 0.0)), reverse=True)
    for ann in annotations:
        category = coco.loadCats([int(ann["category_id"])])[0]
        name = str(category["name"])
        coco_category_id = int(ann["category_id"])
        color = category_colors.get(coco_category_id)
        if color is None:
            continue
        mask = coco.annToMask(ann).astype(bool)
        if mask.shape != (height, width):
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
            mask = np.asarray(
                mask_image.resize((width, height), Image.Resampling.NEAREST)
            ) > 0
        condition[mask] = np.asarray(color, dtype=np.uint8)
        records.append(
            {
                "coco_category": name,
                "coco_category_id": coco_category_id,
                "control_color": color,
                "area": int(mask.sum()),
            }
        )
    return Image.fromarray(condition, mode="RGB"), records


def make_six_panel(original: Image.Image, generated: list[Image.Image], path: Path) -> None:
    tile = 512
    label_h = 50
    canvas = Image.new("RGB", (tile * 3, (tile + label_h) * 2), "white")
    images = [original, *generated]
    labels = ["Original", "Caption 1", "Caption 2", "Caption 3", "Caption 4", "Caption 5"]
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (image, label) in enumerate(zip(images, labels)):
        row, col = divmod(idx, 3)
        x, y = col * tile, row * (tile + label_h)
        canvas.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 12, y + tile + 15), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "_status.json").write_text(
        json.dumps({"stage": "generating", "samples": args.samples}, indent=2),
        encoding="utf-8",
    )
    items = unique_test_items(args.samples)
    coco_by_split = {
        split: COCO(str(args.annotation_dir / f"instances_{split}.json"))
        for split in ("train2017", "val2017")
    }
    category_metadata = json.loads(
        (args.model_root / "panoptic_coco_categories.json").read_text(encoding="utf-8")
    )
    category_colors = {
        int(category["id"]): [int(value) for value in category["color"]]
        for category in category_metadata
    }

    controlnet = ControlNetModel.from_pretrained(
        args.model_root / "lllyasviel__control_v11p_sd15_seg",
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.model_root / "stable-diffusion-v1-5__stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
        safety_checker=None,
        local_files_only=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe = pipe.to(device)

    manifest: list[dict[str, Any]] = []
    for sample_index, item in enumerate(items):
        metadata = item["metadata"]
        original_array = item["image"].cpu().numpy().astype(np.uint8)
        if original_array.ndim == 3 and original_array.shape[0] == 3:
            original_array = np.transpose(original_array, (1, 2, 0))
        original = Image.fromarray(original_array, mode="RGB")
        condition, objects = make_segment_condition(
            coco_by_split[str(metadata["coco_split"])],
            int(metadata["coco_id"]),
            original.size,
            category_colors,
        )
        condition_512 = condition.resize((512, 512), Image.Resampling.NEAREST)
        condition_path = args.output_root / "conditions" / f"{sample_index:03d}_segment.png"
        condition_path.parent.mkdir(parents=True, exist_ok=True)
        condition_512.save(condition_path)

        captions = [
            caption
            for caption, valid in zip(item["captions"], item["caption_mask"])
            if valid and caption.strip()
        ]
        if not captions:
            raise RuntimeError(f"No caption for sample {sample_index}.")
        captions = (captions + [captions[-1]] * 5)[:5]
        generated: list[Image.Image] = []
        for caption_index, caption in enumerate(captions):
            generator = torch.Generator(device=device).manual_seed(
                args.seed + sample_index * 100 + caption_index
            )
            result = pipe(
                prompt=f"{caption}, natural photograph, realistic",
                negative_prompt="drawing, painting, text, watermark, collage, duplicate, distorted, low quality",
                image=condition_512,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_scale,
                generator=generator,
                height=512,
                width=512,
            ).images[0]
            generated.append(result)

        name = (
            f"{sample_index:03d}_label{int(metadata['label_index']):05d}"
            "_six_panel.png"
        )
        relative_file = f"gallery/test/{name}"
        make_six_panel(original, generated, args.output_root / relative_file)
        manifest.append(
            {
                "split": "test",
                "file": relative_file,
                "condition_file": f"conditions/{condition_path.name}",
                "captions": captions,
                "objects": objects,
                **metadata,
            }
        )
        (args.output_root / "_status.json").write_text(
            json.dumps(
                {
                    "stage": "generating",
                    "completed_samples": sample_index + 1,
                    "samples": args.samples,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    config = {
        "name": "Oracle COCO Segmentation + 5 Ground-truth Captions",
        "method": "ControlNet 1.1 segmentation; native COCO category colors and instance masks",
        "samples": args.samples,
        "images_per_sample": 5,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "controlnet_scale": args.controlnet_scale,
        "seed": args.seed,
        "gallery_layout": "Original, Caption 1, Caption 2, Caption 3, Caption 4, Caption 5",
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output_root / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (args.output_root / "_status.json").write_text(
        json.dumps({"stage": "complete", **config}, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
