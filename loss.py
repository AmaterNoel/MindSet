from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class BrainConceptLossConfig:
    global_weight: float = 1.0
    concept_weight: float = 1.0
    consistency_weight: float = 0.2
    sparsity_weight: float = 0.01
    global_temperature: float = 0.07
    concept_temperature: float = 0.07
    confidence_temperature: float = 1.0
    eps: float = 1e-6


def masked_logsumexp(values: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    neg_inf = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask, neg_inf)
    out = torch.logsumexp(masked, dim=dim)
    empty = ~mask.any(dim=dim)
    return out.masked_fill(empty, torch.log(torch.as_tensor(eps, device=values.device, dtype=values.dtype)))


def positive_caption_alignment_loss(
    pred_global: torch.Tensor,
    caption_features: torch.Tensor,
    caption_mask: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_global = F.normalize(pred_global, dim=-1)
    caption_features = F.normalize(caption_features, dim=-1)
    sim = torch.einsum("bd,bkd->bk", pred_global, caption_features)
    scaled = sim / temperature
    soft_best = temperature * masked_logsumexp(scaled, caption_mask, dim=1, eps=eps)
    loss = (1.0 - soft_best).mean()

    safe_sim = sim.masked_fill(~caption_mask, -1e4)
    best_sim = safe_sim.max(dim=1).values
    valid_counts = caption_mask.sum(dim=1).clamp_min(1)
    mean_pos_sim = (sim * caption_mask.float()).sum(dim=1) / valid_counts
    return loss, {
        "global_soft_best_sim": soft_best.detach().mean(),
        "global_best_sim": best_sim.detach().mean(),
        "global_mean_caption_sim": mean_pos_sim.detach().mean(),
    }


def positive_concept_mil_loss(
    query_clip: torch.Tensor,
    query_confidence_logits: torch.Tensor,
    concept_features: torch.Tensor,
    concept_mask: torch.Tensor,
    temperature: float = 0.07,
    confidence_temperature: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    query_clip = F.normalize(query_clip, dim=-1)
    concept_features = F.normalize(concept_features, dim=-1)
    sim = torch.einsum("bqd,bnd->bqn", query_clip, concept_features)
    concept_mask_q = concept_mask[:, None, :].expand_as(sim)

    concept_soft_best_per_query = temperature * masked_logsumexp(
        sim / temperature,
        concept_mask_q,
        dim=2,
        eps=eps,
    )
    log_conf = F.logsigmoid(query_confidence_logits / confidence_temperature)
    query_scores = (concept_soft_best_per_query + log_conf) / temperature
    sample_score = temperature * torch.logsumexp(query_scores, dim=1)
    loss = (1.0 - sample_score).mean()

    safe_sim = sim.masked_fill(~concept_mask_q, -1e4)
    best_sim_per_query = safe_sim.max(dim=2).values
    confidence = torch.sigmoid(query_confidence_logits)
    return loss, {
        "concept_sample_score": sample_score.detach().mean(),
        "concept_best_query_sim": best_sim_per_query.max(dim=1).values.detach().mean(),
        "concept_confidence_mean": confidence.detach().mean(),
        "concept_active_count_05": (confidence > 0.5).float().sum(dim=1).detach().mean(),
    }


def concept_global_consistency_loss(
    pred_global: torch.Tensor,
    query_clip: torch.Tensor,
    query_confidence_logits: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_global = F.normalize(pred_global, dim=-1)
    query_clip = F.normalize(query_clip, dim=-1)
    weights = torch.sigmoid(query_confidence_logits).clamp_min(eps)
    set_feature = (query_clip * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    set_feature = F.normalize(set_feature, dim=-1)
    sim = (set_feature * pred_global).sum(dim=-1)
    loss = (1.0 - sim).mean()
    return loss, {"concept_global_sim": sim.detach().mean()}


def confidence_sparsity_loss(query_confidence_logits: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    confidence = torch.sigmoid(query_confidence_logits)
    loss = confidence.mean()
    return loss, {
        "confidence_mean": confidence.detach().mean(),
        "active_count_05": (confidence > 0.5).float().sum(dim=1).detach().mean(),
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
        global_loss, global_metrics = positive_caption_alignment_loss(
            outputs["global_clip"],
            batch["caption_clip_features"],
            batch["caption_mask"],
            temperature=cfg.global_temperature,
            eps=cfg.eps,
        )
        concept_loss, concept_metrics = positive_concept_mil_loss(
            outputs["query_clip"],
            outputs["query_confidence_logits"],
            batch["concept_clip_features"],
            batch["concept_mask"],
            temperature=cfg.concept_temperature,
            confidence_temperature=cfg.confidence_temperature,
            eps=cfg.eps,
        )
        consistency_loss, consistency_metrics = concept_global_consistency_loss(
            outputs["global_clip"],
            outputs["query_clip"],
            outputs["query_confidence_logits"],
            eps=cfg.eps,
        )
        sparsity_loss, sparsity_metrics = confidence_sparsity_loss(outputs["query_confidence_logits"])

        total = (
            cfg.global_weight * global_loss
            + cfg.concept_weight * concept_loss
            + cfg.consistency_weight * consistency_loss
            + cfg.sparsity_weight * sparsity_loss
        )

        metrics_tensors = {
            "total_loss": total.detach(),
            "global_loss": global_loss.detach(),
            "concept_loss": concept_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "sparsity_loss": sparsity_loss.detach(),
            **global_metrics,
            **concept_metrics,
            **consistency_metrics,
            **sparsity_metrics,
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
