from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import AlexNet_Weights, alexnet

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import (
    DEFAULT_ANNOTATION_DIR,
    DEFAULT_BETAS_1D_PATH,
    DEFAULT_CAPTION_EMBEDDINGS_PATH,
    DEFAULT_IMAGE_EMBEDDINGS_PATH,
    DEFAULT_STIM_INFO_PATH,
    DEFAULT_STIMULUS_H5_PATH,
    collate_nsd_concepts,
    create_dataloaders,
)
from models.train_base_model_1D import AdapterLayer, ResMLP, maybe_pool_fmri
from models.train_low_model_1D import (
    RepeatAveragedDataset,
    batch_images_to_float,
    compute_image_metrics,
    make_repeat_averaged_loaders,
    save_fixed_split_reconstructions,
    save_jsonl,
    set_seed,
)


@dataclass
class StructuredConfig:
    in_dim: int
    hidden_dim: int = 4096
    n_blocks: int = 4
    dropout: float = 0.5
    adapter_bottleneck: int = 128
    seed_channels: int = 128
    image_embedding_dim: int = 1280
    semantic_guidance: bool = False
    reliability_gate: bool = False


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        groups = min(16, out_channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.net(x)


class PyramidRGBDecoder(nn.Module):
    def __init__(self, seed_channels: int, semantic_dim: int = 0) -> None:
        super().__init__()
        self.channels = [seed_channels, 96, 64, 48]
        self.blocks = nn.ModuleList(
            [
                ResidualConvBlock(seed_channels, seed_channels),
                ResidualConvBlock(seed_channels, 96),
                ResidualConvBlock(96, 64),
                ResidualConvBlock(64, 48),
            ]
        )
        self.rgb_heads = nn.ModuleList([nn.Conv2d(channels, 3, 3, padding=1) for channels in self.channels])
        self.modulators = (
            nn.ModuleList([nn.Linear(semantic_dim, channels * 2) for channels in self.channels])
            if semantic_dim > 0
            else None
        )

    def modulate(self, x: torch.Tensor, semantic: torch.Tensor | None, index: int) -> torch.Tensor:
        if semantic is None or self.modulators is None:
            return x
        gamma, beta = self.modulators[index](semantic).chunk(2, dim=-1)
        return x * (1.0 + 0.1 * gamma[:, :, None, None]) + 0.1 * beta[:, :, None, None]

    def forward(
        self,
        seed: torch.Tensor,
        semantic: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = seed
        logits: torch.Tensor | None = None
        outputs: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            if index > 0:
                features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
            features = self.modulate(block(features), semantic, index)
            residual = self.rgb_heads[index](features)
            logits = residual if logits is None else F.interpolate(
                logits, size=residual.shape[-2:], mode="bilinear", align_corners=False
            ) + residual
            outputs.append(torch.sigmoid(logits))
        return outputs[-1], outputs


class StructuredLowModel(nn.Module):
    def __init__(self, config: StructuredConfig, reliability: torch.Tensor | None = None) -> None:
        super().__init__()
        self.config = config
        self.embedder = nn.Sequential(
            AdapterLayer(config.in_dim, config.adapter_bottleneck),
            nn.Linear(config.in_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.translator = ResMLP(config.hidden_dim, config.n_blocks, config.dropout)
        self.seed_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.seed_channels * 8 * 8),
        )
        self.semantic_head = (
            nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.image_embedding_dim),
            )
            if config.semantic_guidance
            else None
        )
        self.decoder = PyramidRGBDecoder(
            config.seed_channels,
            semantic_dim=config.image_embedding_dim if config.semantic_guidance else 0,
        )
        if config.reliability_gate:
            if reliability is None or reliability.numel() != config.in_dim:
                raise ValueError("Reliability gate requires one reliability value per input voxel.")
            reliability = reliability.float().clamp(0.05, 1.0)
            reliability = reliability / reliability.mean().clamp_min(1e-6)
            self.register_buffer("voxel_reliability", reliability)
            self.learnable_voxel_gate = nn.Parameter(torch.zeros(config.in_dim))
        else:
            self.register_buffer("voxel_reliability", torch.ones(config.in_dim), persistent=False)
            self.learnable_voxel_gate = None

    def forward(self, fmri: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if self.learnable_voxel_gate is not None:
            fmri = fmri * self.voxel_reliability * (2.0 * torch.sigmoid(self.learnable_voxel_gate))
        hidden = self.translator(self.embedder(fmri))
        semantic = self.semantic_head(hidden) if self.semantic_head is not None else None
        if self.training and semantic is not None:
            keep = (torch.rand(semantic.shape[0], 1, device=semantic.device) >= 0.3).to(semantic.dtype)
            decoder_semantic = semantic * keep
        else:
            decoder_semantic = semantic
        seed = self.seed_head(hidden).view(fmri.shape[0], self.config.seed_channels, 8, 8)
        rgb, pyramid = self.decoder(seed, decoder_semantic)
        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "low_level_rgb": rgb,
            "pyramid_rgb": pyramid,
        }
        if semantic is not None:
            result["semantic_embedding"] = semantic
        return result


class ImagePyramidEncoder(nn.Module):
    def __init__(self, seed_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            ResidualConvBlock(64, 64),
            nn.Conv2d(64, 96, 4, stride=2, padding=1),
            nn.SiLU(),
            ResidualConvBlock(96, 96),
            nn.Conv2d(96, seed_channels, 4, stride=2, padding=1),
            ResidualConvBlock(seed_channels, seed_channels),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class AlexNetShallow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.features = model.features[:6]
        self.requires_grad_(False)
        self.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image, size=(128, 128), mode="bilinear", align_corners=False)
        return self.features((image - self.mean) / self.std)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu_x = F.avg_pool2d(pred, 7, stride=1, padding=3)
    mu_y = F.avg_pool2d(target, 7, stride=1, padding=3)
    sigma_x = F.avg_pool2d(pred * pred, 7, stride=1, padding=3) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, 7, stride=1, padding=3) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred * target, 7, stride=1, padding=3) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-6)
    return 1.0 - score.mean()


def structured_loss(
    outputs: dict[str, torch.Tensor | list[torch.Tensor]],
    batch: dict[str, Any],
    perceptual_teacher: AlexNetShallow | None,
    multiscale_weight: float,
    gradient_weight: float,
    ssim_weight: float,
    perceptual_weight: float,
    semantic_weight: float,
) -> dict[str, torch.Tensor]:
    pred = outputs["low_level_rgb"]
    assert isinstance(pred, torch.Tensor)
    target = batch_images_to_float(batch["image"], pred.device)
    target = F.interpolate(target, size=pred.shape[-2:], mode="area")
    pyramid = outputs["pyramid_rgb"]
    assert isinstance(pyramid, list)
    pyramid_weights = [0.1, 0.15, 0.25, 0.5]
    pyramid_l1 = pred.new_tensor(0.0)
    for weight, level in zip(pyramid_weights, pyramid, strict=True):
        level_target = F.interpolate(target, size=level.shape[-2:], mode="area")
        pyramid_l1 = pyramid_l1 + weight * F.smooth_l1_loss(level, level_target)
    pixel_mse = F.mse_loss(pred, target)
    grad = gradient_loss(pred, target)
    structure = ssim_loss(pred, target)
    perceptual = pred.new_tensor(0.0)
    if perceptual_teacher is not None:
        with torch.no_grad():
            teacher_target = perceptual_teacher(target)
        perceptual = F.smooth_l1_loss(perceptual_teacher(pred), teacher_target)
    semantic = pred.new_tensor(0.0)
    semantic_pred = outputs.get("semantic_embedding")
    if isinstance(semantic_pred, torch.Tensor):
        semantic_target = batch["image_embeddings"].to(device=pred.device, dtype=semantic_pred.dtype)
        semantic = 1.0 - F.cosine_similarity(semantic_pred, semantic_target, dim=-1).mean()
    loss_pyramid = multiscale_weight * pyramid_l1
    loss_mse = 0.1 * pixel_mse
    loss_gradient = gradient_weight * grad
    loss_ssim = ssim_weight * structure
    loss_perceptual = perceptual_weight * perceptual
    loss_semantic = semantic_weight * semantic
    total = loss_pyramid + loss_mse + loss_gradient + loss_ssim + loss_perceptual + loss_semantic
    return {
        "loss": total,
        "pyramid_l1": pyramid_l1.detach(),
        "pixel_mse": pixel_mse.detach(),
        "gradient": grad.detach(),
        "ssim_loss": structure.detach(),
        "perceptual": perceptual.detach(),
        "semantic": semantic.detach(),
        "loss_pyramid": loss_pyramid.detach(),
        "loss_mse": loss_mse.detach(),
        "loss_gradient": loss_gradient.detach(),
        "loss_ssim": loss_ssim.detach(),
        "loss_perceptual": loss_perceptual.detach(),
        "loss_semantic": loss_semantic.detach(),
    }


def update_meter(total: dict[str, float], values: dict[str, torch.Tensor], count: int) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + float(value.item()) * count


def finish_meter(total: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in total.items()}


def run_epoch(
    model: StructuredLowModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    perceptual_teacher: AlexNetShallow | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if perceptual_teacher is not None:
        perceptual_teacher.eval()
    total: dict[str, float] = {}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            fmri = maybe_pool_fmri(
                batch["fmri"].to(device),
                enable_pool=args.enable_pool,
                pool_num=args.pool_num,
                pool_type=args.pool_type,
            )
            outputs = model(fmri)
            losses = structured_loss(
                outputs,
                batch,
                perceptual_teacher,
                multiscale_weight=args.multiscale_weight,
                gradient_weight=args.gradient_weight,
                ssim_weight=args.ssim_weight,
                perceptual_weight=args.perceptual_weight,
                semantic_weight=args.semantic_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if not training:
                pred = outputs["low_level_rgb"]
                assert isinstance(pred, torch.Tensor)
                target = batch_images_to_float(batch["image"], device)
                losses.update(compute_image_metrics(pred, target))
            batch_count = int(fmri.shape[0])
            update_meter(total, losses, batch_count)
            count += batch_count
    return finish_meter(total, count)


def compute_repeat_reliability(dataset: RepeatAveragedDataset, max_groups: int = 4096) -> torch.Tensor:
    base = dataset.dataset
    groups = [group for group in dataset.groups if len(group) >= 2][:max_groups]
    if not groups:
        return torch.ones_like(dataset.voxel_mean)
    sum_mean = np.zeros(base.betas.shape[1], dtype=np.float64)
    sum_mean_sq = np.zeros(base.betas.shape[1], dtype=np.float64)
    sum_noise = np.zeros(base.betas.shape[1], dtype=np.float64)
    for group in groups:
        rows = [
            int(base.records[base.indices[logical_index]]["beta_row"])
            for logical_index in group[:3]
        ]
        repeats = np.asarray(base.betas[rows], dtype=np.float32)
        mean = repeats.mean(axis=0, dtype=np.float64)
        sum_mean += mean
        sum_mean_sq += np.square(mean)
        sum_noise += repeats.var(axis=0, dtype=np.float64)
    n = len(groups)
    signal = np.maximum(sum_mean_sq / n - np.square(sum_mean / n), 0.0)
    noise = np.maximum(sum_noise / n, 1e-6)
    reliability = signal / (signal + noise / 2.0)
    return torch.from_numpy(reliability.astype(np.float32)).clamp(0.0, 1.0)


def pretrain_decoder(
    model: StructuredLowModel,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    run_dir: Path,
) -> None:
    if epochs <= 0:
        return
    encoder = ImagePyramidEncoder(model.config.seed_channels).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(model.decoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    path = run_dir / "pretrain_metrics.jsonl"
    for epoch in range(1, epochs + 1):
        encoder.train()
        model.decoder.train()
        total = 0.0
        count = 0
        start = time.perf_counter()
        for batch in train_loader:
            image = batch_images_to_float(batch["image"], device)
            image = F.interpolate(image, size=(64, 64), mode="area")
            seed = encoder(image)
            pred, pyramid = model.decoder(seed)
            loss = pred.new_tensor(0.0)
            for weight, level in zip([0.1, 0.15, 0.25, 0.5], pyramid, strict=True):
                target = F.interpolate(image, size=level.shape[-2:], mode="area")
                loss = loss + weight * F.smooth_l1_loss(level, target)
            loss = loss + 0.1 * F.mse_loss(pred, image) + 0.1 * gradient_loss(pred, image)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * image.shape[0]
            count += int(image.shape[0])
        payload = {
            "epoch": epoch,
            "train": {"loss": total / max(count, 1)},
            "train_time_sec": time.perf_counter() - start,
        }
        save_jsonl(path, payload)
        print(f"decoder_pretrain={epoch:03d} loss={payload['train']['loss']:.6f}")
    del encoder, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured low-level fMRI-to-RGB experiments G-K.")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--stim-info-path", type=Path, default=DEFAULT_STIM_INFO_PATH)
    parser.add_argument("--betas-1d-path", type=Path, default=DEFAULT_BETAS_1D_PATH)
    parser.add_argument("--caption-embeddings-path", type=Path, default=DEFAULT_CAPTION_EMBEDDINGS_PATH)
    parser.add_argument("--image-embeddings-path", type=Path, default=DEFAULT_IMAGE_EMBEDDINGS_PATH)
    parser.add_argument("--stimulus-h5-path", type=Path, default=DEFAULT_STIMULUS_H5_PATH)
    parser.add_argument("--output-root", type=Path, default=Path("output/low_model_structured"))
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--adapter-bottleneck", type=int, default=128)
    parser.add_argument("--seed-channels", type=int, default=128)
    parser.add_argument("--enable-pool", type=str2bool, default=True)
    parser.add_argument("--pool-num", type=int, default=8192)
    parser.add_argument("--pool-type", choices=["max", "avg"], default="max")
    parser.add_argument("--semantic-guidance", type=str2bool, default=False)
    parser.add_argument("--reliability-gate", type=str2bool, default=False)
    parser.add_argument("--perceptual-weight", type=float, default=0.0)
    parser.add_argument("--semantic-weight", type=float, default=0.0)
    parser.add_argument("--multiscale-weight", type=float, default=1.0)
    parser.add_argument("--gradient-weight", type=float, default=0.1)
    parser.add_argument("--ssim-weight", type=float, default=0.1)
    parser.add_argument("--decoder-pretrain-epochs", type=int, default=0)
    parser.add_argument("--decoder-pretrain-lr", type=float, default=2e-4)
    parser.add_argument("--fixed-samples-per-split", type=int, default=5)
    parser.add_argument("--delete-checkpoints-after-run", type=str2bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    run_name = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        annotation_dir=args.annotation_dir,
        stim_info_path=args.stim_info_path,
        betas_1d_path=args.betas_1d_path,
        caption_embeddings_path=args.caption_embeddings_path,
        image_embeddings_path=args.image_embeddings_path,
        stimulus_h5_path=args.stimulus_h5_path,
        fmri_format="1d",
        subject=args.subject,
        seed=args.seed,
        normalize="none",
        include_raw=True,
    )
    train_loader, val_loader, test_loader, voxel_mean, voxel_std = make_repeat_averaged_loaders(
        train_loader,
        val_loader,
        test_loader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    first_batch = next(iter(train_loader))
    raw_in_dim = int(first_batch["fmri"].shape[-1])
    model_in_dim = args.pool_num if args.enable_pool else raw_in_dim
    reliability = None
    if args.reliability_gate:
        reliability = compute_repeat_reliability(train_loader.dataset)
        if args.enable_pool:
            reliability = F.adaptive_avg_pool1d(reliability[None, None], args.pool_num).flatten()
        np.save(run_dir / "voxel_reliability.npy", reliability.numpy())
    config = StructuredConfig(
        in_dim=model_in_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        adapter_bottleneck=args.adapter_bottleneck,
        seed_channels=args.seed_channels,
        image_embedding_dim=int(first_batch["image_embeddings"].shape[-1]),
        semantic_guidance=args.semantic_guidance,
        reliability_gate=args.reliability_gate,
    )
    model = StructuredLowModel(config, reliability=reliability).to(device)
    perceptual_teacher = AlexNetShallow().to(device) if args.perceptual_weight > 0 else None
    config_payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": asdict(config),
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(run_dir / "voxel_stats.npz", mean=voxel_mean.numpy(), std=voxel_std.numpy())
    print(f"run_dir={run_dir}")
    print(f"model params={config_payload['parameters']:,}")
    pretrain_decoder(
        model,
        train_loader,
        device,
        epochs=args.decoder_pretrain_epochs,
        lr=args.decoder_pretrain_lr,
        run_dir=run_dir,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = -float("inf")
    best_epoch = -1
    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, optimizer, perceptual_teacher, args)
        payload: dict[str, Any] = {
            "epoch": epoch,
            "train": train_metrics,
            "train_time_sec": time.perf_counter() - start,
        }
        message = f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_start = time.perf_counter()
            val_metrics = run_epoch(model, val_loader, device, None, perceptual_teacher, args)
            payload["val"] = val_metrics
            payload["val_time_sec"] = time.perf_counter() - val_start
            score = 0.5 * (val_metrics["pixcorr"] + val_metrics["ssim"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": asdict(config),
                        "epoch": epoch,
                        "best_metric": "visual",
                        "best_score": best_score,
                    },
                    run_dir / "best_low_model.pt",
                )
            message += (
                f" val_loss={val_metrics['loss']:.6f}"
                f" pixcorr={val_metrics['pixcorr']:.4f}"
                f" ssim={val_metrics['ssim']:.4f}"
                f" psnr={val_metrics['psnr']:.2f}"
                f" best_visual={best_score:.6f}@{best_epoch}"
            )
        print(message)
        save_jsonl(metrics_path, payload)
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": asdict(config), "epoch": args.epochs},
        run_dir / "last_low_model.pt",
    )
    best_path = run_dir / "best_low_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, device, None, perceptual_teacher, args)
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, ensure_ascii=False, indent=2)
    manifest = save_fixed_split_reconstructions(
        model,
        {"train": train_loader.dataset, "val": val_loader.dataset, "test": test_loader.dataset},
        run_dir,
        device,
        enable_pool=args.enable_pool,
        pool_num=args.pool_num,
        pool_type=args.pool_type,
        image_size=256,
        sample_count=args.fixed_samples_per_split,
        vae=None,
    )
    print("saved fixed reconstructions " + " ".join(f"{key}={len(value)}" for key, value in manifest.items()))
    print(
        f"test_loss={test_metrics['loss']:.6f} pixcorr={test_metrics['pixcorr']:.4f} "
        f"ssim={test_metrics['ssim']:.4f} psnr={test_metrics['psnr']:.2f}"
    )
    if args.delete_checkpoints_after_run:
        deleted = []
        for path in [run_dir / "best_low_model.pt", run_dir / "last_low_model.pt"]:
            if path.exists():
                path.unlink()
                deleted.append(path.name)
        print(f"deleted checkpoints after evaluation: {deleted}")
    print(f"saved outputs to {run_dir}")


if __name__ == "__main__":
    main()
