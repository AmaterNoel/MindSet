from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.freesurfer import io as fsio
from scipy.ndimage import map_coordinates
from torch import nn

from dataloader import (
    DEFAULT_BETAS_PATH,
    DEFAULT_FEATURE_PATH,
    DEFAULT_LABEL_PATH,
    DEFAULT_NSD_ROOT,
    NSDConceptDataset,
    collate_nsd_concepts,
    create_dataloaders,
)
from loss import BrainConceptLoss, BrainConceptLossConfig
from model import BrainConceptModelConfig, BrainToConceptModel


PROJECT_ROOT = Path(r"D:\PycharmProjects\MindKeyAnimator")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_SURFACE_ROOT = DEFAULT_NSD_ROOT / "surface"


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class MetricAverager:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, float], n: int) -> None:
        self.count += n
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * n

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.totals.items()}


def safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9._()-]+", "_", text.strip())
    return text[:max_len] if text else "item"


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def print_loss_line(prefix: str, metrics: dict[str, float]) -> None:
    keys = ["total_loss", "global_loss", "concept_loss", "anchor_loss", "confidence_loss", "budget_loss"]
    body = " ".join(f"{key}={metrics.get(key, 0.0):.4f}" for key in keys)
    print(f"{prefix} {body}")


def print_eval_line(prefix: str, metrics: dict[str, float]) -> None:
    print_loss_line(prefix, metrics)
    global_keys = ["global_best_caption_cos", "global_mean_caption_cos", "global_top1_acc", "global_mrr"]
    concept_keys = [
        "concept_active_count",
        "concept_active_best_cos",
        "concept_sample_hit_rate",
        "concept_query_diversity",
    ]
    print(f"{prefix} global " + " ".join(f"{key}={metrics.get(key, 0.0):.4f}" for key in global_keys))
    print(f"{prefix} concept " + " ".join(f"{key}={metrics.get(key, 0.0):.4f}" for key in concept_keys))


def compute_eval_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    confidence_threshold: float,
    concept_hit_threshold: float,
) -> dict[str, float]:
    global_clip = F.normalize(outputs["global_clip"], dim=-1)
    captions = F.normalize(batch["caption_clip_features"], dim=-1)
    caption_mask = batch["caption_mask"].bool()
    cap_sim = torch.einsum("bd,bkd->bk", global_clip, captions)
    cap_sim_masked = cap_sim.masked_fill(~caption_mask, -1e4)
    best_caption = cap_sim_masked.max(dim=1).values
    mean_caption = (cap_sim * caption_mask.float()).sum(dim=1) / caption_mask.sum(dim=1).clamp_min(1)

    # Batch retrieval metric over all valid captions. Same nsd_id is treated as positive.
    flat_caps = captions.reshape(-1, captions.shape[-1])
    flat_mask = caption_mask.reshape(-1)
    flat_caps = flat_caps[flat_mask]
    owners: list[int] = []
    for item_idx, meta in enumerate(batch["metadata"]):
        owners.extend([int(meta["nsd_id"])] * int(caption_mask[item_idx].sum().item()))
    owner_tensor = torch.as_tensor(owners, device=global_clip.device)
    sim_all = global_clip @ flat_caps.T
    nsd_ids = torch.as_tensor([int(meta["nsd_id"]) for meta in batch["metadata"]], device=global_clip.device)
    top_owner = owner_tensor[sim_all.argmax(dim=1)]
    top1_acc = (top_owner == nsd_ids).float().mean()

    mrr_values = []
    order = torch.argsort(sim_all, dim=1, descending=True)
    for i in range(sim_all.shape[0]):
        positive = owner_tensor[order[i]] == nsd_ids[i]
        rank = torch.where(positive)[0]
        mrr_values.append(1.0 / float(rank[0].item() + 1) if rank.numel() else 0.0)
    global_mrr = float(np.mean(mrr_values)) if mrr_values else 0.0

    query_clip = F.normalize(outputs["query_clip"], dim=-1)
    concept_features = F.normalize(batch["concept_clip_features"], dim=-1)
    concept_mask = batch["concept_mask"].bool()
    confidence = torch.sigmoid(outputs["query_confidence_logits"])
    active = confidence > confidence_threshold
    concept_sim = torch.einsum("bqd,bnd->bqn", query_clip, concept_features)
    concept_sim = concept_sim.masked_fill(~concept_mask[:, None, :], -1e4)
    best_per_query = concept_sim.max(dim=2).values

    active_counts = active.float().sum(dim=1)
    active_best_values = []
    sample_hits = []
    diversities = []
    for i in range(query_clip.shape[0]):
        active_idx = torch.where(active[i])[0]
        if active_idx.numel() == 0:
            active_best_values.append(0.0)
            sample_hits.append(0.0)
            diversities.append(0.0)
            continue
        vals = best_per_query[i, active_idx]
        active_best_values.append(float(vals.mean().detach().cpu()))
        sample_hits.append(float((vals >= concept_hit_threshold).any().detach().cpu()))
        if active_idx.numel() > 1:
            q = query_clip[i, active_idx]
            pair = q @ q.T
            tri = torch.triu_indices(pair.shape[0], pair.shape[1], offset=1, device=pair.device)
            diversities.append(float((1.0 - pair[tri[0], tri[1]]).mean().detach().cpu()))
        else:
            diversities.append(0.0)

    return {
        "global_best_caption_cos": float(best_caption.mean().detach().cpu()),
        "global_mean_caption_cos": float(mean_caption.mean().detach().cpu()),
        "global_top1_acc": float(top1_acc.detach().cpu()),
        "global_mrr": global_mrr,
        "concept_active_count": float(active_counts.mean().detach().cpu()),
        "concept_active_best_cos": float(np.mean(active_best_values)) if active_best_values else 0.0,
        "concept_sample_hit_rate": float(np.mean(sample_hits)) if sample_hits else 0.0,
        "concept_query_diversity": float(np.mean(diversities)) if diversities else 0.0,
    }


def parse_freesurfer_patch(path: Path, pial_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords, faces = fsio.read_geometry(str(pial_path))
    with path.open("rb") as fp:
        header = np.frombuffer(fp.read(4), dtype=">i4")[0]
        if int(header) != -1:
            raise ValueError(f"Unexpected FreeSurfer patch header {header} in {path}")
        nverts = int(np.frombuffer(fp.read(4), dtype=">i4")[0])
        data = np.frombuffer(fp.read(), dtype=[("vert", ">i4"), ("x", ">f4"), ("y", ">f4"), ("z", ">f4")])
    if len(data) != nverts:
        raise ValueError(f"Patch vertex count mismatch: expected {nverts}, got {len(data)}")
    ids = np.abs(data["vert"].astype(np.int64)) - 1
    flat = np.zeros_like(coords, dtype=np.float32)
    flat[ids, 0] = data["y"].astype(np.float32)
    flat[ids, 1] = -data["x"].astype(np.float32)
    flat[ids, 2] = data["z"].astype(np.float32)
    xy = flat[:, :2].copy()
    flat[:, 0] = -xy[:, 1]
    flat[:, 1] = xy[:, 0]
    valid_vertices = np.zeros(len(coords), dtype=bool)
    valid_vertices[ids] = True
    valid_faces = faces[valid_vertices[faces].all(axis=1)]
    return flat[:, :2], valid_faces, valid_vertices


class NSDFlatmapProjector:
    def __init__(self, surface_root: Path = DEFAULT_SURFACE_ROOT, layer: str = "layerB2") -> None:
        self.surface_root = Path(surface_root)
        self.layer = layer
        self.hemis: dict[str, dict[str, Any]] = {}
        for hemi in ["lh", "rh"]:
            surf_dir = self.surface_root / "freesurfer" / "subj01" / "surf"
            label_dir = self.surface_root / "freesurfer" / "subj01" / "label"
            xy, faces, valid = parse_freesurfer_patch(
                surf_dir / f"{hemi}.full.flat.patch.3d",
                surf_dir / f"{hemi}.pial",
            )
            transform = np.asarray(
                nib.load(str(self.surface_root / "transforms" / f"{hemi}.func1pt8-to-{layer}.mgz")).dataobj
            ).squeeze()
            labels = []
            for name in ["Kastner2015", "floc-faces", "floc-bodies", "floc-places"]:
                p = label_dir / f"{hemi}.{name}.mgz"
                if p.exists():
                    labels.append(np.asarray(nib.load(str(p)).dataobj).squeeze().astype(np.int32))
            self.hemis[hemi] = {"xy": xy, "faces": faces, "valid": valid, "transform": transform, "labels": labels}

    def project_volume(self, volume: np.ndarray) -> dict[str, np.ndarray]:
        projected = {}
        for hemi, info in self.hemis.items():
            coords = info["transform"].T
            vals = map_coordinates(volume, coords, order=1, mode="nearest")
            projected[hemi] = np.asarray(vals, dtype=np.float32)
        return projected

    def save_flatmap(self, volume: np.ndarray, path: Path, title: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        projected = self.project_volume(volume)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=150)
        finite_vals = np.concatenate([np.abs(v[np.isfinite(v)]) for v in projected.values()])
        vmax = float(np.percentile(finite_vals, 99.5)) if finite_vals.size else 1.0
        vmax = max(vmax, 1e-8)
        for ax, hemi in zip(axes, ["lh", "rh"]):
            info = self.hemis[hemi]
            xy = info["xy"]
            valid = info["valid"]
            faces = info["faces"]
            vals = np.nan_to_num(np.abs(projected[hemi]), nan=0.0, posinf=0.0, neginf=0.0)
            ax.scatter(xy[valid, 0], xy[valid, 1], s=0.05, c="#eeeeee", linewidths=0, rasterized=True)
            tri = mtri.Triangulation(xy[:, 0], xy[:, 1], faces)
            for labels in info["labels"]:
                for lab in np.unique(labels[valid]):
                    if lab <= 0:
                        continue
                    mask = labels == lab
                    if int((mask & valid).sum()) < 30:
                        continue
                    try:
                        ax.tricontour(tri, mask.astype(np.float32), levels=[0.5], colors="#777777", linewidths=0.35)
                    except Exception:
                        pass
            threshold = max(vmax * 0.15, float(np.percentile(vals[valid], 97)) if np.any(vals[valid] > 0) else vmax)
            active = valid & (vals >= threshold)
            if np.any(active):
                ax.scatter(
                    xy[active, 0],
                    xy[active, 1],
                    c=vals[active],
                    cmap="Reds",
                    vmin=threshold,
                    vmax=vmax,
                    s=1.2,
                    linewidths=0,
                    rasterized=True,
                )
            ax.set_aspect("equal")
            ax.axis("off")
        fig.suptitle(title, fontsize=11)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92, wspace=0.01)
        fig.savefig(path)
        plt.close(fig)


def gradient_saliency(
    model: BrainToConceptModel,
    fmri: torch.Tensor,
    score_fn: Any,
    device: torch.device,
) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    x = fmri[:1].to(device).detach().clone().requires_grad_(True)
    outputs = model(x)
    score = score_fn(outputs)
    score.backward()
    sal = (x.grad.detach()[0, 0].abs() * x.detach()[0, 0].abs()).cpu().numpy()
    sal = sal / max(float(np.nanmax(sal)), 1e-8)
    return sal.astype(np.float32)


def visualize_samples(
    model: BrainToConceptModel,
    dataset: NSDConceptDataset,
    out_dir: Path,
    device: torch.device,
    projector: NSDFlatmapProjector,
    max_samples: int,
    confidence_threshold: float,
    max_visual_concepts: int,
) -> None:
    model.eval()
    vocab_features = F.normalize(dataset.concept_clip_features.to(device), dim=-1)
    for sample_idx in range(min(max_samples, len(dataset))):
        item = dataset[sample_idx]
        sample_dir = out_dir / f"sample_{sample_idx:02d}_trial_{item['metadata']['global_trial']:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "metadata": item["metadata"],
                    "target_concepts": item["concept_names"],
                    "captions": item["captions"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        fmri = item["fmri"].unsqueeze(0)
        captions = item["caption_clip_features"].unsqueeze(0).to(device)
        caption_mask = item["caption_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(fmri.to(device))
            global_sim = torch.einsum("bd,bkd->bk", outputs["global_clip"], captions)
            global_sim = global_sim.masked_fill(~caption_mask, -1e4)
            best_caption_idx = int(global_sim.argmax(dim=1).item())
            confidence = torch.sigmoid(outputs["query_confidence_logits"][0])
            active = torch.where(confidence >= confidence_threshold)[0]
            if active.numel() == 0:
                active = torch.topk(confidence, k=1).indices
            active = active[torch.argsort(confidence[active], descending=True)]
            if max_visual_concepts > 0:
                active = active[:max_visual_concepts]
            nearest = torch.argmax(F.normalize(outputs["query_clip"][0, active], dim=-1) @ vocab_features.T, dim=1)

        global_target = captions[0, best_caption_idx].detach()
        global_sal = gradient_saliency(
            model,
            fmri,
            lambda out: torch.dot(F.normalize(out["global_clip"][0], dim=-1), global_target),
            device,
        )
        projector.save_flatmap(global_sal, sample_dir / "global_semantic.png", "global_semantic")

        predictions = []
        for query_idx, vocab_idx in zip(active.tolist(), nearest.tolist()):
            concept_name = dataset.concepts[vocab_idx]
            target = vocab_features[vocab_idx].detach()
            sal = gradient_saliency(
                model,
                fmri,
                lambda out, q=query_idx, t=target: torch.dot(F.normalize(out["query_clip"][0, q], dim=-1), t),
                device,
            )
            conf = float(confidence[query_idx].detach().cpu())
            predictions.append({"query": query_idx + 1, "concept": concept_name, "confidence": conf})
            filename = f"concept_{query_idx + 1:03d}_{safe_name(concept_name)}.png"
            projector.save_flatmap(sal, sample_dir / filename, f"{concept_name}({query_idx + 1:02d})")
        (sample_dir / "predicted_concepts.json").write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@torch.no_grad()
def evaluate(
    model: BrainToConceptModel,
    loader: torch.utils.data.DataLoader,
    criterion: BrainConceptLoss,
    device: torch.device,
    confidence_threshold: float,
    concept_hit_threshold: float,
) -> dict[str, float]:
    model.eval()
    avg = MetricAverager()
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["fmri"])
        _, loss_metrics = criterion(outputs, batch)
        eval_metrics = compute_eval_metrics(outputs, batch, confidence_threshold, concept_hit_threshold)
        metrics = {**loss_metrics, **eval_metrics}
        avg.update(metrics, int(batch["fmri"].shape[0]))
    return avg.compute()


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "metrics": metrics}, path)


def train(args: argparse.Namespace) -> None:
    run_name = datetime.now().strftime("%m%d%H%M")
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = (output_dir / "train.log").open("w", encoding="utf-8")
    with contextlib.redirect_stdout(Tee(sys.stdout, log_file)), contextlib.redirect_stderr(Tee(sys.stderr, log_file)):
        print(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str))
        device = torch.device(args.device)
        train_loader, val_loader, test_loader = create_dataloaders(
            label_path=args.label_path,
            feature_path=args.feature_path,
            betas_path=args.betas_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            group_key=args.group_key,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            normalize=args.normalize,
        )
        print(f"dataset train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

        model = BrainToConceptModel(
            BrainConceptModelConfig(
                base_channels=args.base_channels,
                token_dim=args.token_dim,
                target_grid=tuple(args.target_grid),
                encoder_depth=args.encoder_depth,
                decoder_depth=args.decoder_depth,
                num_heads=args.num_heads,
                dropout=args.dropout,
                num_queries=args.num_queries,
            )
        ).to(device)
        criterion = BrainConceptLoss(
            BrainConceptLossConfig(
                global_weight=args.global_weight,
                concept_weight=args.concept_weight,
                budget_weight=args.budget_weight,
                global_temperature=args.global_temperature,
                semantic_alpha=args.semantic_alpha,
                label_tau=args.label_tau,
                label_scale=args.label_scale,
                query_tau=args.query_tau,
                query_scale=args.query_scale,
                min_label_weight=args.min_label_weight,
                confidence_weight=args.confidence_weight,
                budget_margin=args.budget_margin,
                min_expected_count=args.min_expected_count,
                max_expected_count=args.max_expected_count,
            )
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        projector = NSDFlatmapProjector(args.surface_root, layer=args.surface_layer)

        best_val = math.inf
        for epoch in range(1, args.epochs + 1):
            model.train()
            avg = MetricAverager()
            for batch in train_loader:
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch["fmri"])
                loss, metrics = criterion(outputs, batch)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                avg.update(metrics, int(batch["fmri"].shape[0]))
            train_metrics = avg.compute()
            print_loss_line(f"epoch {epoch:04d} train", train_metrics)

            if epoch % args.val_interval == 0 or epoch == args.epochs:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    args.confidence_threshold,
                    args.concept_hit_threshold,
                )
                print_eval_line(f"epoch {epoch:04d} val", val_metrics)
                val_dir = output_dir / f"val_{epoch:04d}"
                visualize_samples(
                    model,
                    val_loader.dataset,
                    val_dir,
                    device,
                    projector,
                    args.visual_samples,
                    args.visual_confidence_threshold,
                    args.max_visual_concepts,
                )
                if val_metrics["total_loss"] < best_val:
                    best_val = val_metrics["total_loss"]
                    save_checkpoint(output_dir / "best_model.pt", model, optimizer, epoch, val_metrics)
            save_checkpoint(output_dir / "latest_model.pt", model, optimizer, epoch, train_metrics)

        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            args.confidence_threshold,
            args.concept_hit_threshold,
        )
        print_eval_line("test", test_metrics)
        test_dir = output_dir / "test"
        visualize_samples(
            model,
            test_loader.dataset,
            test_dir,
            device,
            projector,
            args.visual_samples,
            args.visual_confidence_threshold,
            args.max_visual_concepts,
        )
        (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    log_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NSD brain-to-global-and-concept CLIP model.")
    parser.add_argument("--nsd-root", type=Path, default=DEFAULT_NSD_ROOT)
    parser.add_argument("--label-path", type=Path, default=None)
    parser.add_argument("--feature-path", type=Path, default=None)
    parser.add_argument("--surface-root", type=Path, default=None)
    parser.add_argument("--betas-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--group-key", default="nsd_id")
    parser.add_argument("--normalize", choices=["none", "volume"], default="volume")

    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--target-grid", type=int, nargs=3, default=[5, 6, 5])
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--decoder-depth", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--num-queries", type=int, default=100)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--global-weight", type=float, default=1.0)
    parser.add_argument("--concept-weight", type=float, default=1.0)
    parser.add_argument("--budget-weight", type=float, default=0.1)
    parser.add_argument("--global-temperature", type=float, default=0.07)
    parser.add_argument("--semantic-alpha", type=float, default=0.6)
    parser.add_argument("--label-tau", type=float, default=0.20)
    parser.add_argument("--label-scale", type=float, default=0.40)
    parser.add_argument("--query-tau", type=float, default=0.25)
    parser.add_argument("--query-scale", type=float, default=0.35)
    parser.add_argument("--min-label-weight", type=float, default=0.20)
    parser.add_argument("--confidence-weight", type=float, default=0.30)
    parser.add_argument("--budget-margin", type=float, default=5.0)
    parser.add_argument("--min-expected-count", type=float, default=3.0)
    parser.add_argument("--max-expected-count", type=float, default=30.0)

    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--concept-hit-threshold", type=float, default=0.30)
    parser.add_argument("--visual-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--visual-samples", type=int, default=5)
    parser.add_argument("--max-visual-concepts", type=int, default=20)
    parser.add_argument("--surface-layer", choices=["pial", "layerB1", "layerB2", "layerB3"], default="layerB2")
    args = parser.parse_args()
    args.label_path = args.label_path or args.nsd_root / "annotations" / "process" / DEFAULT_LABEL_PATH.name
    args.feature_path = args.feature_path or args.nsd_root / "annotations" / "process" / DEFAULT_FEATURE_PATH.name
    args.surface_root = args.surface_root or args.nsd_root / "surface"
    args.betas_path = args.betas_path or args.nsd_root / "subj01" / DEFAULT_BETAS_PATH.name
    return args


if __name__ == "__main__":
    train(parse_args())
