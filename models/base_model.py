from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class BaseBrainModelConfig:
    input_channels: int = 1
    base_channels: int = 24
    token_dim: int = 256
    target_grid: tuple[int, int, int] = (5, 6, 5)
    transformer_depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.15
    embedding_dim: int = 1280
    lowlevel_latent_channels: int = 4
    lowlevel_latent_size: int = 32
    lowlevel_hidden_dim: int = 1024


@dataclass
class BaseEmbeddingLossConfig:
    image_mse_weight: float = 1000.0
    image_soft_clip_weight: float = 0.5
    caption_best_cos_weight: float = 2.0
    caption_best_soft_clip_weight: float = 0.5
    lowlevel_l1_weight: float = 1.0
    soft_clip_temp: float = 0.005
    eps: float = 1e-8


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BrainStem3d(nn.Module):
    def __init__(self, config: BaseBrainModelConfig) -> None:
        super().__init__()
        c = config.base_channels
        self.conv = nn.Sequential(
            ConvBlock3d(config.input_channels, c, kernel_size=5, stride=2),
            ConvBlock3d(c, c * 2, kernel_size=3, stride=2),
            ConvBlock3d(c * 2, c * 4, kernel_size=3, stride=2),
            ConvBlock3d(c * 4, config.token_dim, kernel_size=3, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool3d(config.target_grid)
        self.proj = nn.Sequential(
            nn.Conv3d(config.token_dim, config.token_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, config.token_dim), num_channels=config.token_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.conv(x)))


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Projector(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseBrainModel(nn.Module):
    """Predicts one global OpenCLIP embedding from a 3D fMRI volume."""

    def __init__(self, config: BaseBrainModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or BaseBrainModelConfig()
        cfg = self.config

        if cfg.token_dim % cfg.num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")

        self.stem = BrainStem3d(cfg)
        token_count = cfg.target_grid[0] * cfg.target_grid[1] * cfg.target_grid[2]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.token_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count + 1, cfg.token_dim))
        self.token_dropout = nn.Dropout(cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.num_heads,
            dim_feedforward=int(cfg.token_dim * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_depth)
        self.encoder_norm = nn.LayerNorm(cfg.token_dim)

        self.image_head = MLPHead(
            in_dim=cfg.token_dim,
            hidden_dim=cfg.token_dim * 2,
            out_dim=cfg.embedding_dim,
            dropout=cfg.dropout,
        )
        self.low_level_head = MLPHead(
            in_dim=cfg.token_dim,
            hidden_dim=cfg.lowlevel_hidden_dim,
            out_dim=cfg.lowlevel_latent_channels * cfg.lowlevel_latent_size * cfg.lowlevel_latent_size,
            dropout=cfg.dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def encode_tokens(self, fmri: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if fmri.ndim == 4:
            fmri = fmri.unsqueeze(1)
        if fmri.ndim != 5:
            raise ValueError(f"Expected fmri shape [B,C,D,H,W] or [B,D,H,W], got {tuple(fmri.shape)}")

        spatial = self.stem(fmri)
        tokens = spatial.flatten(2).transpose(1, 2).contiguous()
        cls = self.cls_token.expand(fmri.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        tokens = self.token_dropout(tokens)
        tokens = self.encoder(tokens)
        tokens = self.encoder_norm(tokens)
        return tokens, spatial

    def forward(self, fmri: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, spatial = self.encode_tokens(fmri)
        hidden = tokens[:, 0]

        embedding_raw = self.image_head(hidden)
        low_level_latent = self.low_level_head(hidden).view(
            fmri.shape[0],
            self.config.lowlevel_latent_channels,
            self.config.lowlevel_latent_size,
            self.config.lowlevel_latent_size,
        )

        return {
            "embedding": F.normalize(embedding_raw, dim=-1),
            "embedding_raw": embedding_raw,
            "low_level_latent": low_level_latent,
            "brain_tokens": tokens,
            "brain_spatial": spatial,
        }


def soft_clip_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    temp: float = 0.005,
) -> torch.Tensor:
    preds = F.normalize(preds, dim=-1)
    targets = F.normalize(targets, dim=-1)
    target_logits = targets @ targets.T / temp
    pred_logits = preds @ targets.T / temp

    target_prob = target_logits.softmax(dim=-1)
    loss_forward = -(pred_logits.log_softmax(dim=-1) * target_prob).sum(dim=-1).mean()
    loss_backward = -(pred_logits.T.log_softmax(dim=-1) * target_prob).sum(dim=-1).mean()
    return (loss_forward + loss_backward) * 0.5


def select_best_caption_targets(
    pred_embedding: torch.Tensor,
    caption_embeddings: torch.Tensor,
    caption_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if caption_embeddings.ndim != 3:
        raise ValueError(
            f"Expected caption_embeddings shape [B,N,D], got {tuple(caption_embeddings.shape)}"
        )
    if pred_embedding.shape[0] != caption_embeddings.shape[0]:
        raise ValueError("pred_embedding and caption_embeddings must have the same batch size.")
    if pred_embedding.shape[-1] != caption_embeddings.shape[-1]:
        raise ValueError("pred_embedding and caption_embeddings must have the same embedding dimension.")

    pred_norm = F.normalize(pred_embedding, dim=-1)
    caption_norm = F.normalize(caption_embeddings, dim=-1)
    sims = torch.einsum("bd,bnd->bn", pred_norm, caption_norm)

    if caption_mask is not None:
        mask = caption_mask.to(device=sims.device, dtype=torch.bool)
        if mask.shape != sims.shape:
            raise ValueError(f"caption_mask shape {tuple(mask.shape)} does not match {tuple(sims.shape)}.")
        if not mask.any(dim=1).all():
            raise ValueError("Each sample must have at least one valid caption.")
        sims = sims.masked_fill(~mask, -torch.inf)

    best_idx = sims.argmax(dim=1)
    batch_idx = torch.arange(caption_embeddings.shape[0], device=caption_embeddings.device)
    best_caption = caption_embeddings[batch_idx, best_idx]
    best_sim = sims[batch_idx, best_idx]
    return best_caption, best_idx, best_sim


def compute_base_embedding_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: BaseEmbeddingLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    cfg = config or BaseEmbeddingLossConfig()
    pred_embedding = outputs["embedding"]
    image_target = batch["image_embeddings"].to(device=pred_embedding.device, dtype=pred_embedding.dtype)
    caption_target = batch["caption_text_embeddings"].to(device=pred_embedding.device, dtype=pred_embedding.dtype)
    caption_mask = batch.get("caption_mask")
    if caption_mask is not None:
        caption_mask = caption_mask.to(device=pred_embedding.device)

    image_target = F.normalize(image_target, dim=-1)
    best_caption, best_caption_idx, best_caption_cos = select_best_caption_targets(
        pred_embedding=pred_embedding,
        caption_embeddings=caption_target,
        caption_mask=caption_mask,
    )
    best_caption = F.normalize(best_caption, dim=-1)

    image_mse = F.mse_loss(pred_embedding, image_target)
    image_soft_clip = soft_clip_loss(pred_embedding, image_target, temp=cfg.soft_clip_temp)
    caption_best_cos = 1.0 - best_caption_cos.mean()
    caption_best_soft_clip = soft_clip_loss(pred_embedding, best_caption, temp=cfg.soft_clip_temp)
    lowlevel_l1 = pred_embedding.new_tensor(0.0)
    if "image_vae_latents" in batch and "low_level_latent" in outputs:
        lowlevel_target = batch["image_vae_latents"].to(
            device=pred_embedding.device,
            dtype=outputs["low_level_latent"].dtype,
        )
        if tuple(outputs["low_level_latent"].shape) != tuple(lowlevel_target.shape):
            raise ValueError(
                f"Predicted low-level latent shape {tuple(outputs['low_level_latent'].shape)} "
                f"does not match target {tuple(lowlevel_target.shape)}."
            )
        lowlevel_l1 = F.l1_loss(outputs["low_level_latent"], lowlevel_target)

    total = (
        cfg.image_mse_weight * image_mse
        + cfg.image_soft_clip_weight * image_soft_clip
        + cfg.caption_best_cos_weight * caption_best_cos
        + cfg.caption_best_soft_clip_weight * caption_best_soft_clip
        + cfg.lowlevel_l1_weight * lowlevel_l1
    )

    return {
        "loss": total,
        "loss_image_mse": image_mse,
        "loss_image_soft_clip": image_soft_clip,
        "loss_caption_best_cos": caption_best_cos,
        "loss_caption_best_soft_clip": caption_best_soft_clip,
        "loss_lowlevel_l1": lowlevel_l1,
        "caption_best_cos_mean": best_caption_cos.mean().detach(),
        "caption_best_idx": best_caption_idx.detach(),
    }


class BaseEmbeddingLoss(nn.Module):
    def __init__(self, config: BaseEmbeddingLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or BaseEmbeddingLossConfig()

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return compute_base_embedding_loss(outputs, batch, self.config)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_base_model(**kwargs: Any) -> BaseBrainModel:
    return BaseBrainModel(BaseBrainModelConfig(**kwargs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the base fMRI-to-embedding model.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--transformer-depth", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=1280)
    parser.add_argument("--lowlevel-latent-size", type=int, default=32)
    args = parser.parse_args()

    model = BaseBrainModel(
        BaseBrainModelConfig(
            base_channels=args.base_channels,
            token_dim=args.token_dim,
            transformer_depth=args.transformer_depth,
            embedding_dim=args.embedding_dim,
            lowlevel_latent_size=args.lowlevel_latent_size,
        )
    ).to(args.device)
    model.train()

    fmri = torch.randn(args.batch_size, 1, 81, 104, 83, device=args.device)
    image_embeddings = torch.randn(args.batch_size, args.embedding_dim, device=args.device)
    caption_text_embeddings = torch.randn(args.batch_size, 5, args.embedding_dim, device=args.device)
    caption_mask = torch.ones(args.batch_size, 5, dtype=torch.bool, device=args.device)
    image_vae_latents = torch.randn(args.batch_size, 4, args.lowlevel_latent_size, args.lowlevel_latent_size, device=args.device)
    batch = {
        "fmri": fmri,
        "image_embeddings": image_embeddings,
        "caption_text_embeddings": caption_text_embeddings,
        "caption_mask": caption_mask,
        "image_vae_latents": image_vae_latents,
    }

    outputs = model(batch["fmri"])
    losses = compute_base_embedding_loss(outputs, batch)
    losses["loss"].backward()

    print(f"parameters={count_parameters(model):,}")
    for key, value in outputs.items():
        print(key, tuple(value.shape), value.dtype)
    for key, value in losses.items():
        if value.ndim == 0:
            print(key, float(value.detach().cpu()))
        else:
            print(key, tuple(value.shape), value.dtype)


if __name__ == "__main__":
    main()
