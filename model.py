from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class BrainConceptModelConfig:
    input_channels: int = 1
    base_channels: int = 32
    token_dim: int = 256
    target_grid: tuple[int, int, int] = (5, 6, 5)
    encoder_depth: int = 4
    decoder_depth: int = 3
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.15
    num_queries: int = 100
    clip_dim: int = 512


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BrainStemProjector(nn.Module):
    def __init__(self, config: BrainConceptModelConfig) -> None:
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
        x = self.conv(x)
        x = self.pool(x)
        return self.proj(x)


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


class BrainToConceptModel(nn.Module):
    def __init__(self, config: BrainConceptModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or BrainConceptModelConfig()
        cfg = self.config

        self.stem = BrainStemProjector(cfg)
        token_count = cfg.target_grid[0] * cfg.target_grid[1] * cfg.target_grid[2]
        self.token_pos_embed = nn.Parameter(torch.zeros(1, token_count, cfg.token_dim))
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_depth)
        self.encoder_norm = nn.LayerNorm(cfg.token_dim)

        self.global_query = nn.Parameter(torch.zeros(1, 1, cfg.token_dim))
        self.global_attn = nn.MultiheadAttention(
            embed_dim=cfg.token_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.global_norm = nn.LayerNorm(cfg.token_dim)
        self.global_head = MLPHead(
            in_dim=cfg.token_dim,
            hidden_dim=cfg.token_dim * 2,
            out_dim=cfg.clip_dim,
            dropout=cfg.dropout,
        )

        self.query_embed = nn.Parameter(torch.zeros(1, cfg.num_queries, cfg.token_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.num_heads,
            dim_feedforward=int(cfg.token_dim * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.decoder_depth)
        self.query_norm = nn.LayerNorm(cfg.token_dim)
        self.query_feature_head = MLPHead(
            in_dim=cfg.token_dim,
            hidden_dim=cfg.token_dim * 2,
            out_dim=cfg.clip_dim,
            dropout=cfg.dropout,
        )
        self.query_confidence_head = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.token_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.token_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.global_query, std=0.02)
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def encode_tokens(self, fmri: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = self.stem(fmri)
        tokens = spatial.flatten(2).transpose(1, 2).contiguous()
        tokens = tokens + self.token_pos_embed[:, : tokens.shape[1]]
        tokens = self.token_dropout(tokens)
        tokens = self.encoder(tokens)
        tokens = self.encoder_norm(tokens)
        return tokens, spatial

    def forward(self, fmri: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, spatial = self.encode_tokens(fmri)
        batch_size = fmri.shape[0]

        global_query = self.global_query.expand(batch_size, -1, -1)
        global_hidden, _ = self.global_attn(global_query, tokens, tokens, need_weights=False)
        global_hidden = self.global_norm(global_hidden[:, 0])
        global_clip = F.normalize(self.global_head(global_hidden), dim=-1)

        query_embed = self.query_embed.expand(batch_size, -1, -1)
        query_hidden = self.query_decoder(query_embed, tokens)
        query_hidden = self.query_norm(query_hidden)
        query_clip = F.normalize(self.query_feature_head(query_hidden), dim=-1)
        query_confidence_logits = self.query_confidence_head(query_hidden).squeeze(-1)

        return {
            "global_clip": global_clip,
            "query_clip": query_clip,
            "query_confidence_logits": query_confidence_logits,
            "brain_tokens": tokens,
            "brain_spatial": spatial,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(**kwargs: Any) -> BrainToConceptModel:
    config = BrainConceptModelConfig(**kwargs)
    return BrainToConceptModel(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test brain-to-concept model.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--decoder-depth", type=int, default=3)
    args = parser.parse_args()

    config = BrainConceptModelConfig(
        base_channels=args.base_channels,
        token_dim=args.token_dim,
        num_queries=args.num_queries,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
    )
    model = BrainToConceptModel(config).to(args.device)
    model.eval()
    x = torch.randn(args.batch_size, 1, 81, 104, 83, device=args.device)
    with torch.no_grad():
        out = model(x)
    print(f"parameters={count_parameters(model):,}")
    for key, value in out.items():
        print(key, tuple(value.shape), value.dtype)


if __name__ == "__main__":
    main()
