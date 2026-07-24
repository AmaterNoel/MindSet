from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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
    DEFAULT_SUBJECT,
    create_dataloaders,
    str2bool,
)


@dataclass
class Mind1DConfig:
    in_dim: int = 15724
    out_dim: int = 1280
    hidden_dim: int = 2048
    n_blocks: int = 2
    adapter_bottleneck: int = 128
    dropout: float = 0.5


@dataclass
class LossConfig:
    image_soft_clip_weight: float = 1.0
    text_soft_clip_weight: float = 1.0
    image_mse_weight: float = 1.0
    text_mse_weight: float = 1.0
    soft_clip_temp: float = 0.005


class AdapterLayer(nn.Module):
    def __init__(self, in_channels: int, bottleneck: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.down_proj = nn.Linear(in_channels, bottleneck)
        self.non_linear = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, in_channels)
        self.dropout = dropout
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=np.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down = self.down_proj(x)
        down = self.non_linear(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        return self.up_proj(down) + x


class ResMLP(nn.Module):
    def __init__(self, dim: int, n_blocks: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for block in self.blocks:
            x = block(x) + residual
            residual = x
        return x


class Mind1D(nn.Module):
    def __init__(self, config: Mind1DConfig) -> None:
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
        self.head_image = nn.Linear(config.hidden_dim, config.out_dim)
        self.head_text = nn.Linear(config.hidden_dim, config.out_dim)

    def forward(self, fmri: torch.Tensor) -> dict[str, torch.Tensor]:
        if fmri.ndim != 2:
            raise ValueError(f"Expected 1D fMRI tensor [B,V], got {tuple(fmri.shape)}.")
        hidden = self.translator(self.embedder(fmri))
        pred_image = self.head_image(hidden)
        pred_text = self.head_text(hidden)
        return {
            "image_embedding": F.normalize(pred_image, dim=-1),
            "text_embedding": F.normalize(pred_text, dim=-1),
            "image_embedding_raw": pred_image,
            "text_embedding_raw": pred_text,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pool_voxels(voxels: torch.Tensor, pool_num: int, pool_type: str) -> torch.Tensor:
    if pool_type == "avg":
        return F.adaptive_avg_pool1d(voxels.unsqueeze(1), pool_num).squeeze(1)
    if pool_type == "max":
        return F.adaptive_max_pool1d(voxels.unsqueeze(1), pool_num).squeeze(1)
    raise ValueError(f"Unsupported pool_type: {pool_type}")


def maybe_pool_fmri(voxels: torch.Tensor, enable_pool: bool, pool_num: int, pool_type: str) -> torch.Tensor:
    if not enable_pool:
        return voxels
    if voxels.ndim != 2:
        raise ValueError(f"Expected fmri tensor shape [B,V], got {tuple(voxels.shape)}.")
    return pool_voxels(voxels, pool_num=pool_num, pool_type=pool_type)


def masked_caption_mean(captions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(device=captions.device, dtype=captions.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (captions * mask_f).sum(dim=1) / denom


def soft_clip_loss(preds: torch.Tensor, targets: torch.Tensor, temp: float = 0.005) -> torch.Tensor:
    preds = F.normalize(preds, dim=-1)
    targets = F.normalize(targets, dim=-1)
    target_logits = targets @ targets.T / temp
    pred_logits = preds @ targets.T / temp
    target_prob = target_logits.softmax(dim=-1)
    loss_forward = -(pred_logits.log_softmax(dim=-1) * target_prob).sum(dim=-1).mean()
    loss_backward = -(pred_logits.T.log_softmax(dim=-1) * target_prob).sum(dim=-1).mean()
    return 0.5 * (loss_forward + loss_backward)


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any], cfg: LossConfig) -> dict[str, torch.Tensor]:
    pred_image = outputs["image_embedding"]
    pred_text = outputs["text_embedding"]
    image_target = F.normalize(batch["image_embeddings"].to(pred_image.device, pred_image.dtype), dim=-1)
    caption_embeddings = batch["caption_text_embeddings"].to(pred_text.device, pred_text.dtype)
    caption_mask = batch["caption_mask"].to(pred_text.device)
    text_target = F.normalize(masked_caption_mean(caption_embeddings, caption_mask), dim=-1)

    image_soft = soft_clip_loss(pred_image, image_target, temp=cfg.soft_clip_temp)
    text_soft = soft_clip_loss(pred_text, text_target, temp=cfg.soft_clip_temp)
    image_mse = F.mse_loss(pred_image, image_target)
    text_mse = F.mse_loss(pred_text, text_target)
    loss = (
        cfg.image_soft_clip_weight * image_soft
        + cfg.text_soft_clip_weight * text_soft
        + cfg.image_mse_weight * image_mse
        + cfg.text_mse_weight * text_mse
    )
    return {
        "loss": loss,
        "image_soft_clip": image_soft.detach(),
        "text_soft_clip": text_soft.detach(),
        "image_mse": image_mse.detach(),
        "text_mse": text_mse.detach(),
    }


def aggregate_meter(total: dict[str, float], values: dict[str, torch.Tensor], n: int) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + float(value.item()) * n


def finalize_meter(total: dict[str, float], n: int) -> dict[str, float]:
    return {key: value / max(n, 1) for key, value in total.items()}


def evaluate_two_way(pred_features: torch.Tensor, candidate_features: torch.Tensor, num_trials: int, seed: int) -> dict[str, float]:
    logits = (pred_features @ candidate_features.T).detach().cpu()
    labels = torch.arange(logits.shape[0], dtype=torch.long)
    rows = torch.arange(labels.numel(), dtype=torch.long)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    acc = []
    for _ in range(num_trials):
        neg = torch.randint(0, labels.numel() - 1, labels.shape, generator=rng)
        neg = neg + (neg >= labels).long()
        pos_scores = logits[rows, labels]
        neg_scores = logits[rows, neg]
        correct = (pos_scores > neg_scores).float()
        tie = (pos_scores == neg_scores).float()
        acc.append((correct + 0.5 * tie).mean().item())
    return {"two_way_mean": float(np.mean(acc)), "two_way_std": float(np.std(acc))}


def retrieval_metrics(pred_features: torch.Tensor, candidate_features: torch.Tensor, two_way_trials: int, seed: int) -> dict[str, float]:
    pred_features = F.normalize(pred_features.float(), dim=-1)
    candidate_features = F.normalize(candidate_features.float(), dim=-1)
    logits = pred_features @ candidate_features.T
    labels = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
    metrics: dict[str, float] = {}
    for k in (1, 5, 10, 50, 100, 200):
        k_eff = min(k, candidate_features.shape[0])
        topk = torch.topk(logits, k=k_eff, dim=1).indices
        metrics[f"top{k}_acc"] = float((topk == labels).any(dim=1).float().mean().item())

    sorted_idx = torch.argsort(logits, dim=1, descending=True)
    rank = sorted_idx.eq(labels).float().argmax(dim=1).float() + 1.0
    cos = F.cosine_similarity(pred_features, candidate_features, dim=1)
    two_way = evaluate_two_way(pred_features, candidate_features, num_trials=two_way_trials, seed=seed)
    metrics.update(
        {
            "two_way_mean": two_way["two_way_mean"],
            "two_way_std": two_way["two_way_std"],
            "cosine_mean": float(cos.mean().item()),
            "cosine_std": float(cos.std(unbiased=False).item()),
            "mean_rank": float(rank.mean().item()),
            "median_rank": float(rank.median().item()),
            "mrr": float((1.0 / rank).mean().item()),
        }
    )
    return metrics


@torch.no_grad()
def collect_predictions(
    model: Mind1D,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    enable_pool: bool,
    pool_num: int,
    pool_type: str,
) -> dict[str, torch.Tensor]:
    model.eval()
    pred_img, pred_txt, targ_img, targ_txt, label_ids = [], [], [], [], []
    for batch in loader:
        fmri = maybe_pool_fmri(
            batch["fmri"].to(device),
            enable_pool=enable_pool,
            pool_num=pool_num,
            pool_type=pool_type,
        )
        outputs = model(fmri)
        image_target = F.normalize(batch["image_embeddings"].to(device=device, dtype=outputs["image_embedding"].dtype), dim=-1)
        text_target = F.normalize(
            masked_caption_mean(
                batch["caption_text_embeddings"].to(device=device, dtype=outputs["text_embedding"].dtype),
                batch["caption_mask"].to(device),
            ),
            dim=-1,
        )
        pred_img.append(outputs["image_embedding"].detach().cpu())
        pred_txt.append(outputs["text_embedding"].detach().cpu())
        targ_img.append(image_target.detach().cpu())
        targ_txt.append(text_target.detach().cpu())
        label_ids.extend(int(item["label_index"]) for item in batch["metadata"])

    return {
        "pred_image": torch.cat(pred_img, dim=0),
        "pred_text": torch.cat(pred_txt, dim=0),
        "target_image": torch.cat(targ_img, dim=0),
        "target_text": torch.cat(targ_txt, dim=0),
        "label_index": torch.tensor(label_ids, dtype=torch.long),
    }


def aggregate_by_label(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    labels = payload["label_index"]
    unique = torch.unique(labels, sorted=True)
    out: dict[str, list[torch.Tensor]] = {
        "pred_image": [],
        "pred_text": [],
        "target_image": [],
        "target_text": [],
    }
    for label in unique:
        mask = labels == label
        out["pred_image"].append(payload["pred_image"][mask].mean(dim=0))
        out["pred_text"].append(payload["pred_text"][mask].mean(dim=0))
        first = int(torch.nonzero(mask, as_tuple=False)[0].item())
        out["target_image"].append(payload["target_image"][first])
        out["target_text"].append(payload["target_text"][first])
    return {key: torch.stack(values, dim=0) for key, values in out.items()}


@torch.no_grad()
def evaluate_model(
    model: Mind1D,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_cfg: LossConfig,
    enable_pool: bool,
    pool_num: int,
    pool_type: str,
    two_way_trials: int,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    total: dict[str, float] = {}
    count = 0
    for batch in loader:
        fmri = maybe_pool_fmri(
            batch["fmri"].to(device),
            enable_pool=enable_pool,
            pool_num=pool_num,
            pool_type=pool_type,
        )
        outputs = model(fmri)
        losses = compute_loss(outputs, {k: v for k, v in batch.items() if k != "metadata"}, loss_cfg)
        n = int(fmri.shape[0])
        aggregate_meter(total, losses, n)
        count += n

    preds = aggregate_by_label(
        collect_predictions(
            model=model,
            loader=loader,
            device=device,
            enable_pool=enable_pool,
            pool_num=pool_num,
            pool_type=pool_type,
        )
    )
    return {
        "losses": finalize_meter(total, count),
        "image": retrieval_metrics(preds["pred_image"], preds["target_image"], two_way_trials=two_way_trials, seed=seed),
        "text": retrieval_metrics(preds["pred_text"], preds["target_text"], two_way_trials=two_way_trials, seed=seed),
        "unique_labels": int(preds["pred_image"].shape[0]),
    }


def train_one_epoch(
    model: Mind1D,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_cfg: LossConfig,
    enable_pool: bool,
    pool_num: int,
    pool_type: str,
) -> dict[str, float]:
    model.train()
    total: dict[str, float] = {}
    count = 0
    for batch in loader:
        fmri = maybe_pool_fmri(
            batch["fmri"].to(device),
            enable_pool=enable_pool,
            pool_num=pool_num,
            pool_type=pool_type,
        )
        outputs = model(fmri)
        losses = compute_loss(outputs, {k: v for k, v in batch.items() if k != "metadata"}, loss_cfg)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()

        n = int(fmri.shape[0])
        aggregate_meter(total, losses, n)
        count += n
    return finalize_meter(total, count)


def save_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-subject 1D NSD fMRI embedding model.")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--stim-info-path", type=Path, default=DEFAULT_STIM_INFO_PATH)
    parser.add_argument("--betas-1d-path", type=Path, default=DEFAULT_BETAS_1D_PATH)
    parser.add_argument("--caption-embeddings-path", type=Path, default=DEFAULT_CAPTION_EMBEDDINGS_PATH)
    parser.add_argument("--image-embeddings-path", type=Path, default=DEFAULT_IMAGE_EMBEDDINGS_PATH)
    parser.add_argument("--stimulus-h5-path", type=Path, default=DEFAULT_STIMULUS_H5_PATH)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output" / "base_model_1D")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--subject", type=int, default=DEFAULT_SUBJECT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--adapter-bottleneck", type=int, default=128)
    parser.add_argument("--enable-pool", type=str2bool, default=True)
    parser.add_argument("--pool-num", type=int, default=8192)
    parser.add_argument("--pool-type", choices=["max", "avg"], default="max")
    parser.add_argument("--normalize", choices=["none", "volume"], default="volume")
    parser.add_argument("--image-soft-clip-weight", type=float, default=1.0)
    parser.add_argument("--text-soft-clip-weight", type=float, default=1.0)
    parser.add_argument("--image-mse-weight", type=float, default=1000.0)
    parser.add_argument("--text-mse-weight", type=float, default=1000.0)
    parser.add_argument("--soft-clip-temp", type=float, default=0.005)
    parser.add_argument("--two-way-trials", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

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
        normalize=args.normalize,
        include_vae_latents=False,
        include_raw=False,
    )
    first_batch = next(iter(train_loader))
    raw_in_dim = int(first_batch["fmri"].shape[-1])
    model_in_dim = args.pool_num if args.enable_pool else raw_in_dim
    out_dim = int(first_batch["image_embeddings"].shape[-1])

    model_cfg = Mind1DConfig(
        in_dim=model_in_dim,
        out_dim=out_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        adapter_bottleneck=args.adapter_bottleneck,
        dropout=args.dropout,
    )
    loss_cfg = LossConfig(
        image_soft_clip_weight=args.image_soft_clip_weight,
        text_soft_clip_weight=args.text_soft_clip_weight,
        image_mse_weight=args.image_mse_weight,
        text_mse_weight=args.text_mse_weight,
        soft_clip_temp=args.soft_clip_temp,
    )
    model = Mind1D(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config_payload = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": asdict(model_cfg),
        "loss": asdict(loss_cfg),
        "raw_in_dim": raw_in_dim,
        "model_in_dim": model_in_dim,
        "out_dim": out_dim,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)
    print(f"run_dir={run_dir}")
    print(f"model params={config_payload['parameters']:,} trainable={config_payload['trainable_parameters']:,}")
    print(f"raw_in_dim={raw_in_dim} model_in_dim={model_in_dim} out_dim={out_dim}")

    best_score = -float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_cfg=loss_cfg,
            enable_pool=args.enable_pool,
            pool_num=args.pool_num,
            pool_type=args.pool_type,
        )
        train_time = time.perf_counter() - start

        payload: dict[str, Any] = {"epoch": epoch, "train": train_losses, "train_time_sec": train_time}
        message = (
            f"epoch={epoch:03d} train_loss={train_losses['loss']:.6f} "
            f"img_soft={train_losses['image_soft_clip']:.4f} txt_soft={train_losses['text_soft_clip']:.4f} "
            f"img_mse={train_losses['image_mse']:.6f} txt_mse={train_losses['text_mse']:.6f} "
            f"time={train_time:.1f}s"
        )

        do_eval = args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs)
        if do_eval:
            eval_start = time.perf_counter()
            val_metrics = evaluate_model(
                model=model,
                loader=val_loader,
                device=device,
                loss_cfg=loss_cfg,
                enable_pool=args.enable_pool,
                pool_num=args.pool_num,
                pool_type=args.pool_type,
                two_way_trials=args.two_way_trials,
                seed=args.seed,
            )
            eval_time = time.perf_counter() - eval_start
            val_score = 0.5 * (val_metrics["image"]["two_way_mean"] + val_metrics["text"]["two_way_mean"])
            payload["val"] = val_metrics
            payload["val_time_sec"] = eval_time
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                torch.save(
                    {"model_state_dict": model.state_dict(), "model_config": asdict(model_cfg), "epoch": epoch},
                    run_dir / "best_model.pt",
                )
            message += (
                f" val_loss={val_metrics['losses']['loss']:.6f}"
                f" img_two_way={val_metrics['image']['two_way_mean']:.4f}"
                f" txt_two_way={val_metrics['text']['two_way_mean']:.4f}"
                f" img_cos={val_metrics['image']['cosine_mean']:.4f}"
                f" txt_cos={val_metrics['text']['cosine_mean']:.4f}"
                f" best={best_score:.4f}@{best_epoch}"
                f" eval_time={eval_time:.1f}s"
            )
        print(message)
        save_jsonl(metrics_path, payload)

    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": asdict(model_cfg), "epoch": args.epochs},
        run_dir / "last_model.pt",
    )
    best_path = run_dir / "best_model.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        loss_cfg=loss_cfg,
        enable_pool=args.enable_pool,
        pool_num=args.pool_num,
        pool_type=args.pool_type,
        two_way_trials=args.two_way_trials,
        seed=args.seed,
    )
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    print(
        f"test image_two_way={test_metrics['image']['two_way_mean']:.4f} "
        f"text_two_way={test_metrics['text']['two_way_mean']:.4f} "
        f"image_cos={test_metrics['image']['cosine_mean']:.4f} "
        f"text_cos={test_metrics['text']['cosine_mean']:.4f}"
    )
    print(f"saved outputs to {run_dir}")


if __name__ == "__main__":
    main()
