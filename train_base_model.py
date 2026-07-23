from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from dataloader import (
    DEFAULT_ANNOTATION_DIR,
    DEFAULT_BETAS_PATH,
    DEFAULT_CAPTION_EMBEDDINGS_PATH,
    DEFAULT_IMAGE_EMBEDDINGS_PATH,
    DEFAULT_IMAGE_VAE_LATENTS_PATH,
    DEFAULT_STIM_INFO_PATH,
    DEFAULT_STIMULUS_H5_PATH,
    collate_nsd_concepts,
    create_dataloaders,
    create_datasets,
    str2bool,
)
from models.base_model import (
    BaseBrainModel,
    BaseBrainModelConfig,
    BaseEmbeddingLossConfig,
    compute_base_embedding_loss,
    count_parameters,
)
from test_generate import load_pipeline, make_ip_adapter_embeds


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "output" / "base_model"
DEFAULT_MODEL_ROOT = ROOT / "save_pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the base fMRI-to-embedding/low-level-latent model.")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--stim-info-path", type=Path, default=DEFAULT_STIM_INFO_PATH)
    parser.add_argument("--betas-path", type=Path, default=DEFAULT_BETAS_PATH)
    parser.add_argument("--caption-embeddings-path", type=Path, default=DEFAULT_CAPTION_EMBEDDINGS_PATH)
    parser.add_argument("--image-embeddings-path", type=Path, default=DEFAULT_IMAGE_EMBEDDINGS_PATH)
    parser.add_argument("--image-vae-latents-path", type=Path, default=DEFAULT_IMAGE_VAE_LATENTS_PATH)
    parser.add_argument("--stimulus-h5-path", type=Path, default=DEFAULT_STIMULUS_H5_PATH)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=5, help="Evaluate every N epochs. Use 0 to evaluate only after the final epoch.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--target-grid", type=str, default="5,6,5")
    parser.add_argument("--transformer-depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--embedding-dim", type=int, default=1280)
    parser.add_argument("--lowlevel-latent-size", type=int, default=32)
    parser.add_argument("--lowlevel-hidden-dim", type=int, default=1024)

    parser.add_argument("--image-mse-weight", type=float, default=1000.0)
    parser.add_argument("--image-soft-clip-weight", type=float, default=0.5)
    parser.add_argument("--caption-best-cos-weight", type=float, default=2.0)
    parser.add_argument("--caption-best-soft-clip-weight", type=float, default=0.5)
    parser.add_argument("--lowlevel-l1-weight", type=float, default=1.0)
    parser.add_argument("--soft-clip-temp", type=float, default=0.005)

    parser.add_argument("--save-recon", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--recon-image-size", type=int, default=256)
    parser.add_argument("--recon-num-inference-steps", type=int, default=20)
    parser.add_argument("--recon-strength", type=float, default=0.8)
    parser.add_argument("--recon-guidance-scale", type=float, default=5.0)
    parser.add_argument("--recon-controlnet-scale", type=float, default=0.55)
    parser.add_argument("--recon-ip-adapter-scale", type=float, default=1.0)
    parser.add_argument("--recon-seed", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="a natural photograph, high detail")
    parser.add_argument("--negative-prompt", type=str, default="low quality, blurry, distorted, text, watermark")
    parser.add_argument("--cpu-offload", type=str2bool, nargs="?", const=True, default=True)
    return parser.parse_args()


def parse_target_grid(value: str) -> tuple[int, int, int]:
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--target-grid must contain three comma-separated integers.")
    return parts[0], parts[1], parts[2]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def should_evaluate_epoch(epoch: int, epochs: int, eval_every: int) -> bool:
    if eval_every < 0:
        raise ValueError("eval_every must be >= 0.")
    return epoch == epochs or (eval_every > 0 and epoch % eval_every == 0)


def make_model_config(args: argparse.Namespace) -> BaseBrainModelConfig:
    return BaseBrainModelConfig(
        base_channels=args.base_channels,
        token_dim=args.token_dim,
        target_grid=parse_target_grid(args.target_grid),
        transformer_depth=args.transformer_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        embedding_dim=args.embedding_dim,
        lowlevel_latent_size=args.lowlevel_latent_size,
        lowlevel_hidden_dim=args.lowlevel_hidden_dim,
    )


def make_loss_config(args: argparse.Namespace) -> BaseEmbeddingLossConfig:
    return BaseEmbeddingLossConfig(
        image_mse_weight=args.image_mse_weight,
        image_soft_clip_weight=args.image_soft_clip_weight,
        caption_best_cos_weight=args.caption_best_cos_weight,
        caption_best_soft_clip_weight=args.caption_best_soft_clip_weight,
        lowlevel_l1_weight=args.lowlevel_l1_weight,
        soft_clip_temp=args.soft_clip_temp,
    )


def move_training_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("fmri", "caption_text_embeddings", "image_embeddings", "caption_mask", "image_vae_latents"):
        if key in moved:
            moved[key] = moved[key].to(device, non_blocking=True)
    return moved


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def update_meter(meter: dict[str, float], losses: dict[str, torch.Tensor], batch_size: int) -> None:
    meter["samples"] = meter.get("samples", 0.0) + float(batch_size)
    for key, value in losses.items():
        if key == "caption_best_idx" or value.ndim != 0:
            continue
        meter[key] = meter.get(key, 0.0) + float(value.detach().cpu()) * batch_size


def finalize_meter(meter: dict[str, float]) -> dict[str, float]:
    samples = max(meter.get("samples", 0.0), 1.0)
    return {key: value / samples for key, value in meter.items() if key != "samples"}


def weighted_loss_terms(metrics: dict[str, float], loss_config: BaseEmbeddingLossConfig) -> dict[str, float]:
    return {
        "w_image_mse": metrics.get("loss_image_mse", 0.0) * loss_config.image_mse_weight,
        "w_image_soft_clip": metrics.get("loss_image_soft_clip", 0.0) * loss_config.image_soft_clip_weight,
        "w_caption_best_cos": metrics.get("loss_caption_best_cos", 0.0) * loss_config.caption_best_cos_weight,
        "w_caption_best_soft_clip": (
            metrics.get("loss_caption_best_soft_clip", 0.0) * loss_config.caption_best_soft_clip_weight
        ),
        "w_lowlevel_l1": metrics.get("loss_lowlevel_l1", 0.0) * loss_config.lowlevel_l1_weight,
    }


def format_loss_line(prefix: str, metrics: dict[str, float], loss_config: BaseEmbeddingLossConfig) -> str:
    weighted = weighted_loss_terms(metrics, loss_config)
    return (
        f"{prefix}_loss={metrics.get('loss', 0.0):.6f} "
        f"w_img_mse={weighted['w_image_mse']:.4f} "
        f"w_img_soft={weighted['w_image_soft_clip']:.4f} "
        f"w_cap_cos={weighted['w_caption_best_cos']:.4f} "
        f"w_cap_soft={weighted['w_caption_best_soft_clip']:.4f} "
        f"w_low_l1={weighted['w_lowlevel_l1']:.4f} "
        f"cap_cos={metrics.get('caption_best_cos_mean', 0.0):.4f}"
    )


def format_duration(seconds: float) -> str:
    minutes, sec = divmod(max(float(seconds), 0.0), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:04.1f}s"
    if minutes:
        return f"{minutes:d}m{sec:04.1f}s"
    return f"{sec:.1f}s"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_config: BaseEmbeddingLossConfig,
    device: torch.device,
    amp_enabled: bool,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    meter: dict[str, float] = {}
    for batch in loader:
        batch = move_training_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            outputs = model(batch["fmri"])
            losses = compute_base_embedding_loss(outputs, batch, loss_config)

        scaler.scale(losses["loss"]).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        update_meter(meter, losses, int(batch["fmri"].shape[0]))
    return finalize_meter(meter)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_config: BaseEmbeddingLossConfig,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    meter: dict[str, float] = {}
    grouped: dict[int, dict[str, Any]] = {}

    for batch in loader:
        batch = move_training_batch(batch, device)
        with autocast_context(device, amp_enabled):
            outputs = model(batch["fmri"])
            losses = compute_base_embedding_loss(outputs, batch, loss_config)
        update_meter(meter, losses, int(batch["fmri"].shape[0]))

        pred_embedding = outputs["embedding"].detach().float().cpu()
        image_target = batch["image_embeddings"].detach().float().cpu()
        caption_target = batch["caption_text_embeddings"].detach().float().cpu()
        caption_mask = batch["caption_mask"].detach().bool().cpu()

        for row, metadata in enumerate(batch["metadata"]):
            label_index = int(metadata["label_index"])
            entry = grouped.setdefault(
                label_index,
                {
                    "pred_embedding": [],
                    "image_target": image_target[row],
                    "caption_target": caption_target[row],
                    "caption_mask": caption_mask[row],
                },
            )
            entry["pred_embedding"].append(pred_embedding[row])

    metrics = finalize_meter(meter)
    metrics.update(compute_retrieval_metrics(grouped))
    return metrics


def compute_retrieval_metrics(grouped: dict[int, dict[str, Any]]) -> dict[str, float]:
    if not grouped:
        return {
            "image_two_way": 0.0,
            "image_cosine": 0.0,
            "text_two_way": 0.0,
            "text_cosine": 0.0,
        }

    label_ids = sorted(grouped)
    pred_embedding = []
    image_target = []
    caption_target = []
    caption_mask = []
    for label_id in label_ids:
        entry = grouped[label_id]
        pred_embedding.append(torch.stack(entry["pred_embedding"]).mean(dim=0))
        image_target.append(entry["image_target"])
        caption_target.append(entry["caption_target"])
        caption_mask.append(entry["caption_mask"])

    pred_embedding_t = F.normalize(torch.stack(pred_embedding), dim=-1)
    image_target_t = F.normalize(torch.stack(image_target), dim=-1)
    caption_target_t = F.normalize(torch.stack(caption_target), dim=-1)
    caption_mask_t = torch.stack(caption_mask).bool()

    image_sim = pred_embedding_t @ image_target_t.T
    text_sim_all = torch.einsum("id,jkd->ijk", pred_embedding_t, caption_target_t)
    text_sim_all = text_sim_all.masked_fill(~caption_mask_t.unsqueeze(0), -torch.inf)
    text_sim = text_sim_all.max(dim=-1).values

    return {
        "image_two_way": two_way_top1(image_sim),
        "image_cosine": float(image_sim.diag().mean()),
        "text_two_way": two_way_top1(text_sim),
        "text_cosine": float(text_sim.diag().mean()),
    }


def two_way_top1(similarity: torch.Tensor) -> float:
    labels = torch.arange(similarity.shape[0])
    forward = (similarity.argmax(dim=1) == labels).float().mean()
    backward = (similarity.argmax(dim=0) == labels).float().mean()
    return float((forward + backward) * 0.5)


def tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got {tuple(image.shape)}")
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    array = image.detach().cpu().numpy()
    if array.dtype != "uint8":
        array = (array.clip(0, 1) * 255).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def save_image_grid(columns: list[tuple[str, Image.Image]], output: Path) -> None:
    panel = 256
    label_h = 28
    gap = 12
    width = panel * len(columns) + gap * (len(columns) - 1)
    height = panel + label_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in columns:
        canvas.paste(image.resize((panel, panel), Image.Resampling.BICUBIC), (x, 0))
        draw.text((x + 6, panel + 7), label, fill=(0, 0, 0))
        x += panel + gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def decode_low_level_latent(pipe: Any, latent: torch.Tensor, image_size: int) -> Image.Image:
    vae = pipe.vae
    vae_dtype = next(vae.parameters()).dtype
    device = latent.device
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    decoded = vae.decode(latent.to(dtype=vae_dtype) / scaling_factor).sample
    decoded = (decoded / 2.0 + 0.5).clamp(0, 1)
    image = tensor_image_to_pil(decoded[0].detach().cpu())
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return image


@torch.no_grad()
def save_validation_reconstruction(
    model: nn.Module,
    val_recon_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> Path:
    model.eval()
    batch = next(iter(val_recon_loader))
    original = tensor_image_to_pil(batch["image"][0])
    fmri = batch["fmri"].to(device, non_blocking=True)

    outputs = model(fmri)
    embedding = outputs["embedding"][0].detach()

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    args.ip_adapter_scale = args.recon_ip_adapter_scale
    conditioned_pipe = load_pipeline(args, dtype)
    low_level_image = decode_low_level_latent(
        pipe=conditioned_pipe,
        latent=outputs["low_level_latent"].detach().to(device),
        image_size=args.recon_image_size,
    )
    ip_adapter_embeds = make_ip_adapter_embeds(
        embedding,
        dtype=dtype,
        do_classifier_free_guidance=args.recon_guidance_scale > 1.0,
    )
    ip_adapter_embeds = [item.to(device=device, dtype=dtype) for item in ip_adapter_embeds]
    semantic_generator = torch.Generator(device=str(device)).manual_seed(args.recon_seed + epoch)

    semantic_only = conditioned_pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=low_level_image,
        control_image=low_level_image,
        ip_adapter_image_embeds=ip_adapter_embeds,
        strength=1.0,
        num_inference_steps=args.recon_num_inference_steps,
        guidance_scale=args.recon_guidance_scale,
        controlnet_conditioning_scale=0.0,
        generator=semantic_generator,
        height=args.recon_image_size,
        width=args.recon_image_size,
    ).images[0]

    conditioned_generator = torch.Generator(device=str(device)).manual_seed(args.recon_seed + epoch)
    conditioned_reconstruction = conditioned_pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=low_level_image,
        control_image=low_level_image,
        ip_adapter_image_embeds=ip_adapter_embeds,
        strength=args.recon_strength,
        num_inference_steps=args.recon_num_inference_steps,
        guidance_scale=args.recon_guidance_scale,
        controlnet_conditioning_scale=args.recon_controlnet_scale,
        generator=conditioned_generator,
        height=args.recon_image_size,
        width=args.recon_image_size,
    ).images[0]

    output = args.run_dir / f"recon_val_{epoch:03d}.png"
    save_image_grid(
        [
            ("Original", original),
            ("Low-level", low_level_image),
            ("Semantic only", semantic_only),
            ("Semantic + low", conditioned_reconstruction),
        ],
        output,
    )
    return output


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    model_config: BaseBrainModelConfig,
    loss_config: BaseEmbeddingLossConfig,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "model_config": asdict(model_config),
            "loss_config": asdict(loss_config),
            "args": vars(args),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def write_metrics_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_dir = args.output_dir / run_name
    args.run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    model_config = make_model_config(args)
    loss_config = make_loss_config(args)

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        annotation_dir=args.annotation_dir,
        stim_info_path=args.stim_info_path,
        betas_path=args.betas_path,
        caption_embeddings_path=args.caption_embeddings_path,
        image_embeddings_path=args.image_embeddings_path,
        image_vae_latents_path=args.image_vae_latents_path,
        stimulus_h5_path=args.stimulus_h5_path,
        include_vae_latents=True,
        include_raw=False,
    )
    _, val_recon_set, _ = create_datasets(
        annotation_dir=args.annotation_dir,
        stim_info_path=args.stim_info_path,
        betas_path=args.betas_path,
        caption_embeddings_path=args.caption_embeddings_path,
        image_embeddings_path=args.image_embeddings_path,
        image_vae_latents_path=args.image_vae_latents_path,
        stimulus_h5_path=args.stimulus_h5_path,
        include_vae_latents=True,
        include_raw=True,
    )
    val_recon_loader = DataLoader(
        val_recon_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_nsd_concepts,
    )

    model = BaseBrainModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(f"device={device} amp={amp_enabled}")
    print(f"parameters={count_parameters(model):,}")
    print(f"train_batches={len(train_loader)} val_batches={len(val_loader)}")
    print(f"run_dir={args.run_dir}")

    best_val_loss = float("inf")
    metrics_log = args.run_dir / "metrics.jsonl"
    config_path = args.run_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "model_config": asdict(model_config),
                "loss_config": asdict(loss_config),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_config=loss_config,
            device=device,
            amp_enabled=amp_enabled,
            grad_clip=args.grad_clip,
        )
        train_seconds = time.perf_counter() - epoch_start
        print(
            f"epoch={epoch:03d} {format_loss_line('train', train_metrics, loss_config)} "
            f"train_time={format_duration(train_seconds)}"
        )

        should_eval = should_evaluate_epoch(epoch, args.epochs, args.eval_every)
        if not should_eval:
            write_metrics_log(
                metrics_log,
                {"epoch": epoch, "split": "train", **train_metrics, **weighted_loss_terms(train_metrics, loss_config)},
            )
            continue

        eval_start = time.perf_counter()
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            loss_config=loss_config,
            device=device,
            amp_enabled=amp_enabled,
        )
        eval_seconds = time.perf_counter() - eval_start
        print(
            f"epoch={epoch:03d} {format_loss_line('val', val_metrics, loss_config)} "
            f"img_two_way={val_metrics['image_two_way']:.4f} img_cos={val_metrics['image_cosine']:.4f} "
            f"text_two_way={val_metrics['text_two_way']:.4f} text_cos={val_metrics['text_cosine']:.4f} "
            f"eval_time={format_duration(eval_seconds)}"
        )

        save_checkpoint(
            path=args.run_dir / "last_base_model.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            model_config=model_config,
            loss_config=loss_config,
            args=args,
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                path=args.run_dir / "best_base_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                model_config=model_config,
                loss_config=loss_config,
                args=args,
            )
            print(f"saved best checkpoint: {args.run_dir / 'best_base_model.pt'}")

        if args.save_recon:
            recon_start = time.perf_counter()
            recon_path = save_validation_reconstruction(
                model=model,
                val_recon_loader=val_recon_loader,
                args=args,
                device=device,
                epoch=epoch,
            )
            recon_seconds = time.perf_counter() - recon_start
            print(f"saved reconstruction: {recon_path} recon_time={format_duration(recon_seconds)}")

        write_metrics_log(
            metrics_log,
            {"epoch": epoch, "split": "train", **train_metrics, **weighted_loss_terms(train_metrics, loss_config)},
        )
        write_metrics_log(
            metrics_log,
            {"epoch": epoch, "split": "val", **val_metrics, **weighted_loss_terms(val_metrics, loss_config)},
        )


if __name__ == "__main__":
    main()
