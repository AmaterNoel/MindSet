from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F


@dataclass
class BrainConceptLossConfig:
    global_weight: float = 1.0
    concept_weight: float = 1.0
    budget_weight: float = 0.1
    global_temperature: float = 0.07
    semantic_alpha: float = 0.6
    label_tau: float = 0.20
    label_scale: float = 0.40
    query_tau: float = 0.25
    query_scale: float = 0.35
    min_label_weight: float = 0.20
    confidence_weight: float = 0.30
    budget_margin: float = 5.0
    min_expected_count: float = 3.0
    max_expected_count: float = 30.0
    confidence_temperature: float = 1.0
    eps: float = 1e-6


def masked_logsumexp(values: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    neg_inf = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask, neg_inf)
    out = torch.logsumexp(masked, dim=dim)
    empty = ~mask.any(dim=dim)
    return out.masked_fill(empty, torch.log(torch.as_tensor(eps, device=values.device, dtype=values.dtype)))


def soft_threshold(value: torch.Tensor, tau: float, scale: float) -> torch.Tensor:
    return ((value - tau) / max(scale, 1e-6)).clamp(0.0, 1.0)


def positive_caption_alignment_loss(
    pred_global: torch.Tensor,
    caption_features: torch.Tensor,
    caption_mask: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    pred_global = F.normalize(pred_global, dim=-1)
    caption_features = F.normalize(caption_features, dim=-1)
    sim = torch.einsum("bd,bkd->bk", pred_global, caption_features)
    scaled = sim / temperature
    soft_best = temperature * masked_logsumexp(scaled, caption_mask, dim=1, eps=eps)
    loss = (1.0 - soft_best).mean()

    safe_sim = sim.masked_fill(~caption_mask, -1e4)
    best_idx = safe_sim.argmax(dim=1)
    best_sim = safe_sim.gather(1, best_idx[:, None]).squeeze(1)
    best_caption = caption_features[torch.arange(caption_features.shape[0], device=caption_features.device), best_idx]
    valid_counts = caption_mask.sum(dim=1).clamp_min(1)
    mean_pos_sim = (sim * caption_mask.float()).sum(dim=1) / valid_counts
    return loss, {
        "global_soft_best_sim": soft_best.detach().mean(),
        "global_best_sim": best_sim.detach().mean(),
        "global_mean_caption_sim": mean_pos_sim.detach().mean(),
    }, best_caption.detach()


def semantic_gated_concept_loss(
    pred_global: torch.Tensor,
    query_clip: torch.Tensor,
    query_confidence_logits: torch.Tensor,
    concept_features: torch.Tensor,
    concept_mask: torch.Tensor,
    caption_global: torch.Tensor,
    cfg: BrainConceptLossConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    pred_global = F.normalize(pred_global, dim=-1)
    query_clip = F.normalize(query_clip, dim=-1)
    concept_features = F.normalize(concept_features, dim=-1)
    caption_global = F.normalize(caption_global, dim=-1)

    mixed_global = F.normalize(
        cfg.semantic_alpha * caption_global + (1.0 - cfg.semantic_alpha) * pred_global.detach(),
        dim=-1,
    ).detach()

    label_global_sim = torch.einsum("bnd,bd->bn", concept_features, mixed_global)
    label_weights = soft_threshold(label_global_sim, cfg.label_tau, cfg.label_scale) * concept_mask.float()
    query_global_sim = torch.einsum("bqd,bd->bq", query_clip.detach(), mixed_global)
    query_global_weights = soft_threshold(query_global_sim, cfg.query_tau, cfg.query_scale)
    query_label_sim = torch.einsum("bqd,bnd->bqn", query_clip, concept_features)

    anchor_losses: list[torch.Tensor] = []
    confidence_losses: list[torch.Tensor] = []
    matched_sims: list[torch.Tensor] = []
    matched_weights: list[torch.Tensor] = []
    matched_counts: list[int] = []
    effective_label_counts: list[torch.Tensor] = []

    for batch_idx in range(query_clip.shape[0]):
        valid = torch.where((concept_mask[batch_idx]) & (label_weights[batch_idx] >= cfg.min_label_weight))[0]
        conf_target = query_global_weights[batch_idx].clone()

        if valid.numel() == 0:
            anchor_losses.append(query_clip.new_tensor(0.0))
            matched_counts.append(0)
            effective_label_counts.append(query_clip.new_tensor(0.0))
        else:
            weights = label_weights[batch_idx, valid]
            sim = query_label_sim[batch_idx, :, valid]
            cost = -(sim.detach() * weights[None, :]).cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            query_idx = torch.as_tensor(row_ind, device=query_clip.device, dtype=torch.long)
            concept_idx = torch.as_tensor(col_ind, device=query_clip.device, dtype=torch.long)

            assigned_sim = sim[query_idx, concept_idx]
            assigned_weight = weights[concept_idx]
            weight_sum = assigned_weight.sum().clamp_min(cfg.eps)
            anchor_losses.append(((1.0 - assigned_sim) * assigned_weight).sum() / weight_sum)
            conf_target[query_idx] = torch.maximum(conf_target[query_idx], assigned_weight.detach())
            matched_sims.append(assigned_sim.detach().mean())
            matched_weights.append(assigned_weight.detach().mean())
            matched_counts.append(int(query_idx.numel()))
            effective_label_counts.append((label_weights[batch_idx] >= cfg.min_label_weight).float().sum().detach())

        logits = query_confidence_logits[batch_idx] / cfg.confidence_temperature
        confidence_losses.append(F.binary_cross_entropy_with_logits(logits, conf_target.detach()))

    anchor_loss = torch.stack(anchor_losses).mean()
    confidence_loss = torch.stack(confidence_losses).mean()
    concept_loss = anchor_loss + cfg.confidence_weight * confidence_loss

    confidence = torch.sigmoid(query_confidence_logits)
    effective_count = torch.stack(effective_label_counts).mean() if effective_label_counts else query_clip.new_tensor(0.0)
    metrics = {
        "anchor_loss": anchor_loss.detach(),
        "confidence_loss": confidence_loss.detach(),
        "label_weight_mean": label_weights[concept_mask].detach().mean() if concept_mask.any() else query_clip.new_tensor(0.0),
        "query_global_weight_mean": query_global_weights.detach().mean(),
        "concept_matched_sim": torch.stack(matched_sims).mean() if matched_sims else query_clip.new_tensor(0.0),
        "concept_matched_weight": torch.stack(matched_weights).mean() if matched_weights else query_clip.new_tensor(0.0),
        "concept_matched_count": query_clip.new_tensor(float(sum(matched_counts) / max(len(matched_counts), 1))),
        "effective_label_count": effective_count,
        "concept_confidence_mean": confidence.detach().mean(),
        "concept_active_count_05": (confidence > 0.5).float().sum(dim=1).detach().mean(),
    }
    return concept_loss, label_weights.detach(), metrics


def query_budget_loss(
    query_confidence_logits: torch.Tensor,
    label_weights: torch.Tensor,
    cfg: BrainConceptLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    confidence = torch.sigmoid(query_confidence_logits)
    effective_labels = (label_weights >= cfg.min_label_weight).float().sum(dim=1)
    expected_count = (effective_labels + cfg.budget_margin).clamp(cfg.min_expected_count, cfg.max_expected_count)
    target_budget = expected_count / query_confidence_logits.shape[1]
    mean_conf = confidence.mean(dim=1)
    loss = ((mean_conf - target_budget) ** 2).mean()
    return loss, {
        "budget_loss": loss.detach(),
        "confidence_mean": confidence.detach().mean(),
        "active_count_05": (confidence > 0.5).float().sum(dim=1).detach().mean(),
        "expected_active_count": expected_count.detach().mean(),
        "target_budget": target_budget.detach().mean(),
    }


class BrainConceptLoss(nn.Module):
    def __init__(self, config: BrainConceptLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or BrainConceptLossConfig()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        global_loss, global_metrics, caption_global = positive_caption_alignment_loss(
            outputs["global_clip"],
            batch["caption_clip_features"],
            batch["caption_mask"],
            temperature=cfg.global_temperature,
            eps=cfg.eps,
        )
        concept_loss, label_weights, concept_metrics = semantic_gated_concept_loss(
            outputs["global_clip"],
            outputs["query_clip"],
            outputs["query_confidence_logits"],
            batch["concept_clip_features"],
            batch["concept_mask"],
            caption_global,
            cfg,
        )
        budget_loss, budget_metrics = query_budget_loss(outputs["query_confidence_logits"], label_weights, cfg)

        total = cfg.global_weight * global_loss + cfg.concept_weight * concept_loss + cfg.budget_weight * budget_loss

        metrics_tensors = {
            "total_loss": total.detach(),
            "global_loss": global_loss.detach(),
            "concept_loss": concept_loss.detach(),
            "budget_loss": budget_loss.detach(),
            **global_metrics,
            **concept_metrics,
            **budget_metrics,
        }
        metrics = {key: float(value.detach().cpu()) for key, value in metrics_tensors.items()}
        return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test brain concept losses.")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--concepts", type=int, default=14)
    parser.add_argument("--captions", type=int, default=5)
    parser.add_argument("--clip-dim", type=int, default=512)
    args = parser.parse_args()

    outputs = {
        "global_clip": F.normalize(torch.randn(args.batch_size, args.clip_dim), dim=-1),
        "query_clip": F.normalize(torch.randn(args.batch_size, args.queries, args.clip_dim), dim=-1),
        "query_confidence_logits": torch.randn(args.batch_size, args.queries),
    }
    batch = {
        "caption_clip_features": F.normalize(torch.randn(args.batch_size, args.captions, args.clip_dim), dim=-1),
        "caption_mask": torch.ones(args.batch_size, args.captions, dtype=torch.bool),
        "concept_clip_features": F.normalize(torch.randn(args.batch_size, args.concepts, args.clip_dim), dim=-1),
        "concept_mask": torch.ones(args.batch_size, args.concepts, dtype=torch.bool),
    }
    criterion = BrainConceptLoss()
    loss, metrics = criterion(outputs, batch)
    print("loss", float(loss))
    for key, value in metrics.items():
        print(key, value)


if __name__ == "__main__":
    main()
