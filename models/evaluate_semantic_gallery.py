from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate semantic-only reconstructions with frozen CLIP.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=ROOT / "save_pt")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = CLIPModel.from_pretrained(
        args.model_root / "openai__clip-vit-base-patch32",
        local_files_only=True,
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        args.model_root / "openai__clip-vit-base-patch32",
        local_files_only=True,
    )
    model.eval()

    results: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for split in ("train", "val", "test"):
            files = sorted((args.run_dir / "gallery" / split).glob("*_comparison.png"))
            originals, reconstructions = [], []
            for path in files:
                image = Image.open(path).convert("RGB")
                originals.append(image.crop((0, 0, 512, 512)))
                reconstructions.append(image.crop((512, 0, 1024, 512)))
            inputs = processor(images=originals + reconstructions, return_tensors="pt")
            pixels = inputs["pixel_values"].to(device)
            vision_output = model.vision_model(pixel_values=pixels)
            features = F.normalize(model.visual_projection(vision_output.pooler_output).float(), dim=-1)
            original_features, reconstruction_features = features[: len(files)], features[len(files) :]
            cosine = F.cosine_similarity(reconstruction_features, original_features, dim=-1)
            logits = reconstruction_features @ original_features.T
            results[split] = {
                "generated_clip_cosine_mean": float(cosine.mean().item()),
                "generated_top1_within_fixed5": float(
                    (logits.argmax(dim=1) == torch.arange(len(files), device=device)).float().mean().item()
                ),
            }

    metrics_path = args.run_dir / "generation_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["generated_clip_b32"] = results
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
