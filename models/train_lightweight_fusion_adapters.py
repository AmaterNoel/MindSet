from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import create_dataloaders  # noqa: E402
from models.generate_s7_g_controlnet import load_models  # noqa: E402
from models.train_base_model_1D import maybe_pool_fmri  # noqa: E402
from models.train_low_model_1D import batch_images_to_float  # noqa: E402
from models.train_low_model_structured import make_repeat_averaged_loaders  # noqa: E402


class SemanticResidualAdapter(nn.Module):
    def __init__(self, dim: int = 1280, hidden: int = 2048) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value + self.net(self.norm(value)), dim=-1)


class SpatialRGBAdapter(nn.Module):
    def __init__(self, width: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return (image + 0.25 * torch.tanh(self.net(image))).clamp(0, 1)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx, target_dx = pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1]
    pred_dy, target_dy = pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s7-checkpoint", type=Path, required=True)
    parser.add_argument("--g-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    s7, g = load_models(args, device)
    s7.requires_grad_(False).eval()
    g.requires_grad_(False).eval()
    semantic = SemanticResidualAdapter().to(device)
    spatial = SpatialRGBAdapter().to(device)
    optimizer = torch.optim.AdamW(
        list(semantic.parameters()) + list(spatial.parameters()), lr=args.lr, weight_decay=1e-4
    )
    loaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fmri_format="1d",
        normalize="none",
        include_raw=True,
    )
    train_loader, val_loader, _, _, _ = make_repeat_averaged_loaders(
        *loaders, batch_size=args.batch_size, num_workers=args.num_workers
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output.parent / "adapter_metrics.jsonl"
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        semantic.train()
        spatial.train()
        total = count = 0
        for batch in train_loader:
            fmri = batch["fmri"].to(device)
            with torch.no_grad():
                s7_input = maybe_pool_fmri(
                    fmri, fmri.shape[-1] != s7.config.in_dim, s7.config.in_dim, "max"
                )
                g_input = maybe_pool_fmri(
                    fmri, fmri.shape[-1] != g.config.in_dim, g.config.in_dim, "max"
                )
                source_semantic = s7(s7_input)["image_embedding"]
                low = g(g_input)["low_level_rgb"]
                target_semantic = F.normalize(
                    batch["image_embeddings"].to(device, source_semantic.dtype), dim=-1
                )
                target_rgb = F.interpolate(
                    batch_images_to_float(batch["image"], device),
                    size=low.shape[-2:],
                    mode="area",
                )
            adapted_semantic = semantic(source_semantic)
            adapted_rgb = spatial(low)
            semantic_cos = 1 - F.cosine_similarity(adapted_semantic, target_semantic, dim=-1).mean()
            semantic_mse = F.mse_loss(adapted_semantic, target_semantic)
            retain = 1 - F.cosine_similarity(adapted_semantic, source_semantic, dim=-1).mean()
            spatial_l1 = F.l1_loss(adapted_rgb, target_rgb)
            spatial_grad = gradient_loss(adapted_rgb, target_rgb)
            loss = semantic_cos + 100.0 * semantic_mse + 0.2 * retain + spatial_l1 + 0.1 * spatial_grad
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(semantic.parameters()) + list(spatial.parameters()), 1.0
            )
            optimizer.step()
            total += float(loss.item()) * fmri.shape[0]
            count += int(fmri.shape[0])
        payload = {"epoch": epoch, "train_loss": total / max(count, 1)}
        if epoch % 2 == 0 or epoch == args.epochs:
            semantic.eval()
            spatial.eval()
            val_total = val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    fmri = batch["fmri"].to(device)
                    source = s7(
                        maybe_pool_fmri(
                            fmri, fmri.shape[-1] != s7.config.in_dim, s7.config.in_dim, "max"
                        )
                    )["image_embedding"]
                    low = g(
                        maybe_pool_fmri(
                            fmri, fmri.shape[-1] != g.config.in_dim, g.config.in_dim, "max"
                        )
                    )["low_level_rgb"]
                    target_sem = F.normalize(
                        batch["image_embeddings"].to(device, source.dtype), dim=-1
                    )
                    target_rgb = F.interpolate(
                        batch_images_to_float(batch["image"], device),
                        size=low.shape[-2:],
                        mode="area",
                    )
                    val_loss = (
                        1 - F.cosine_similarity(semantic(source), target_sem, dim=-1).mean()
                        + F.l1_loss(spatial(low), target_rgb)
                    )
                    val_total += float(val_loss.item()) * fmri.shape[0]
                    val_count += int(fmri.shape[0])
            payload["val_loss"] = val_total / max(val_count, 1)
            if payload["val_loss"] < best:
                best = payload["val_loss"]
                torch.save(
                    {
                        "semantic_adapter": semantic.state_dict(),
                        "spatial_adapter": spatial.state_dict(),
                        "epoch": epoch,
                        "val_loss": best,
                    },
                    args.output,
                )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
