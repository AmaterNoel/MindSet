from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import Decoder
from PIL import Image, ImageDraw
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_base,
    convnext_small,
    convnext_tiny,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import (  # noqa: E402
    DEFAULT_ANNOTATION_DIR,
    DEFAULT_BETAS_1D_PATH,
    DEFAULT_CAPTION_EMBEDDINGS_PATH,
    DEFAULT_IMAGE_EMBEDDINGS_PATH,
    DEFAULT_IMAGE_VAE_LATENTS_PATH,
    DEFAULT_SPLIT_SEED,
    DEFAULT_STIM_INFO_PATH,
    DEFAULT_STIMULUS_H5_PATH,
    DEFAULT_SUBJECT,
    collate_nsd_concepts,
    create_dataloaders,
    str2bool,
)
from models.train_base_model_1D import AdapterLayer, ResMLP, maybe_pool_fmri  # noqa: E402


@dataclass
class Low1DConfig:
    in_dim: int = 15724
    hidden_dim: int = 4096
    n_blocks: int = 4
    adapter_bottleneck: int = 128
    dropout: float = 0.5
    image_embedding_dim: int = 1280
    spatial_channels: int = 64
    spatial_seed_size: int = 8
    aux_dim: int = 768
    output_channels: int = 3
    output_size: int = 64
    target_mode: str = "rgb"
    input_rank: int = 0
    latent_channels: int = 4
    latent_size: int = 32


@dataclass
class LowLossConfig:
    low_l1_weight: float = 1.0
    low_mse_weight: float = 0.1
    aux_cont_weight: float = 0.1
    image_mse_weight: float = 0.0
    image_soft_clip_weight: float = 0.0
    soft_clip_temp: float = 0.005
    aux_cont_temp: float = 0.2
    aux_cont_max_tokens: int = 4096
    latent_l1_weight: float = 1.0
    decoded_l1_weight: float = 0.0
    decoded_edge_weight: float = 0.0
    decoded_loss_samples: int = 8


def dataclass_from_payload(cls: type, payload: dict[str, Any]) -> Any:
    names = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in names})


class Low1DModel(nn.Module):
    def __init__(self, config: Low1DConfig) -> None:
        super().__init__()
        self.config = config
        if config.target_mode not in {"rgb", "vae"}:
            raise ValueError(f"Unsupported target_mode={config.target_mode!r}.")
        if config.target_mode == "rgb" and (
            config.output_size < config.spatial_seed_size or config.output_size % config.spatial_seed_size != 0
        ):
            raise ValueError(
                "output_size must be a multiple of spatial_seed_size: "
                f"output_size={config.output_size}, spatial_seed_size={config.spatial_seed_size}"
            )
        if config.input_rank > 0:
            self.embedder = nn.Sequential(
                nn.Linear(config.in_dim, config.input_rank, bias=False),
                nn.LayerNorm(config.input_rank),
                nn.GELU(),
                nn.Linear(config.input_rank, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        else:
            self.embedder = nn.Sequential(
                AdapterLayer(config.in_dim, config.adapter_bottleneck),
                nn.Linear(config.in_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        self.translator = ResMLP(config.hidden_dim, config.n_blocks, config.dropout)
        if config.target_mode == "vae":
            self.latent_head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.latent_channels * config.latent_size * config.latent_size),
            )
            return
        seed_dim = config.spatial_channels * config.spatial_seed_size * config.spatial_seed_size
        self.low_seed = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, seed_dim),
        )
        self.seed_norm = nn.GroupNorm(1, config.spatial_channels)
        self.b_aux_projector = nn.Sequential(
            nn.Conv2d(config.spatial_channels, config.aux_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, config.aux_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.aux_dim, config.aux_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, config.aux_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.aux_dim, config.aux_dim, kernel_size=1, bias=True),
        )
        self.low_decoder = Decoder(
            in_channels=config.spatial_channels,
            out_channels=config.output_channels,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[32, 64, 128, 128],
            layers_per_block=1,
        )

    def forward(self, fmri: torch.Tensor) -> dict[str, torch.Tensor]:
        if fmri.ndim != 2:
            raise ValueError(f"Expected 1D fMRI tensor [B,V], got {tuple(fmri.shape)}.")
        hidden = self.translator(self.embedder(fmri))
        if self.config.target_mode == "vae":
            latent = self.latent_head(hidden).view(
                fmri.shape[0],
                self.config.latent_channels,
                self.config.latent_size,
                self.config.latent_size,
            )
            return {"vae_latent": latent}
        seed = self.low_seed(hidden).view(
            fmri.shape[0],
            self.config.spatial_channels,
            self.config.spatial_seed_size,
            self.config.spatial_seed_size,
        )
        low_feature = self.seed_norm(seed)
        b_aux = self.b_aux_projector(low_feature).flatten(2).permute(0, 2, 1).contiguous()
        rgb = torch.sigmoid(self.low_decoder(low_feature))
        return {"low_level_rgb": rgb, "b_aux": b_aux}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_train_voxel_stats(dataset: Any, chunk_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [int(dataset.records[idx]["beta_row"]) for idx in dataset.indices]
    total = np.zeros(dataset.betas.shape[1], dtype=np.float64)
    total_sq = np.zeros(dataset.betas.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(rows), chunk_size):
        chunk = np.asarray(dataset.betas[rows[start : start + chunk_size]], dtype=np.float32)
        total += chunk.sum(axis=0, dtype=np.float64)
        total_sq += np.square(chunk, dtype=np.float32).sum(axis=0, dtype=np.float64)
        count += int(chunk.shape[0])
    mean = total / max(count, 1)
    variance = np.maximum(total_sq / max(count, 1) - np.square(mean), 1e-6)
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(np.sqrt(variance).astype(np.float32))


class RepeatAveragedDataset(Dataset):
    def __init__(
        self,
        dataset: Any,
        voxel_mean: torch.Tensor,
        voxel_std: torch.Tensor,
        random_subsets: bool,
    ) -> None:
        self.dataset = dataset
        self.voxel_mean = voxel_mean
        self.voxel_std = voxel_std
        self.random_subsets = random_subsets
        grouped: dict[int, list[int]] = {}
        for logical_index, record_index in enumerate(dataset.indices):
            nsd_id = int(dataset.records[record_index]["nsd_id"])
            grouped.setdefault(nsd_id, []).append(logical_index)
        self.groups = list(grouped.values())

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self.groups[index]
        selected = group
        if self.random_subsets and len(group) > 1:
            selected = random.sample(group, k=random.randint(1, len(group)))
        item = self.dataset[group[0]]
        volumes = []
        for logical_index in selected:
            record_index = self.dataset.indices[logical_index]
            beta_row = int(self.dataset.records[record_index]["beta_row"])
            volumes.append(torch.from_numpy(self.dataset._load_fmri(beta_row).copy()).float())
        fmri = torch.stack(volumes).mean(dim=0)
        item["fmri"] = (fmri - self.voxel_mean) / self.voxel_std
        item["metadata"] = {
            **item["metadata"],
            "repeat_count": len(selected),
            "available_repeats": len(group),
            "group_index": index,
        }
        return item


def make_repeat_averaged_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    mean, std = compute_train_voxel_stats(train_loader.dataset)
    datasets = [
        RepeatAveragedDataset(train_loader.dataset, mean, std, random_subsets=True),
        RepeatAveragedDataset(val_loader.dataset, mean, std, random_subsets=False),
        RepeatAveragedDataset(test_loader.dataset, mean, std, random_subsets=False),
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=index == 0,
            num_workers=num_workers,
            collate_fn=collate_nsd_concepts,
            pin_memory=torch.cuda.is_available(),
        )
        for index, dataset in enumerate(datasets)
    ]
    return loaders[0], loaders[1], loaders[2], mean, std


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def soft_clip_loss(preds: torch.Tensor, targets: torch.Tensor, temp: float = 0.005) -> torch.Tensor:
    preds = F.normalize(preds.float(), dim=-1)
    targets = F.normalize(targets.float(), dim=-1)
    target_logits = targets @ targets.T / temp
    pred_logits = preds @ targets.T / temp
    loss_forward = -(pred_logits.log_softmax(dim=-1) * target_logits.softmax(dim=-1)).sum(dim=-1).mean()
    loss_backward = -(pred_logits.T.log_softmax(dim=-1) * target_logits.softmax(dim=-1)).sum(dim=-1).mean()
    return (loss_forward + loss_backward) * 0.5


def soft_cont_loss(
    student_preds: torch.Tensor,
    teacher_preds: torch.Tensor,
    teacher_aug_preds: torch.Tensor,
    temp: float = 0.2,
    max_tokens: int = 4096,
) -> torch.Tensor:
    student_preds = F.normalize(student_preds.float().reshape(-1, student_preds.shape[-1]), dim=-1)
    teacher_preds = F.normalize(teacher_preds.float().reshape(-1, teacher_preds.shape[-1]), dim=-1)
    teacher_aug_preds = F.normalize(teacher_aug_preds.float().reshape(-1, teacher_aug_preds.shape[-1]), dim=-1)
    if max_tokens > 0 and student_preds.shape[0] > max_tokens:
        index = torch.randperm(student_preds.shape[0], device=student_preds.device)[:max_tokens]
        student_preds = student_preds[index]
        teacher_preds = teacher_preds[index]
        teacher_aug_preds = teacher_aug_preds[index]

    teacher_teacher_aug = teacher_preds @ teacher_aug_preds.T / temp
    teacher_teacher_aug_t = teacher_aug_preds @ teacher_preds.T / temp
    student_teacher_aug = student_preds @ teacher_aug_preds.T / temp
    student_teacher_aug_t = teacher_aug_preds @ student_preds.T / temp

    loss1 = -(student_teacher_aug.log_softmax(dim=-1) * teacher_teacher_aug.softmax(dim=-1)).sum(dim=-1).mean()
    loss2 = -(student_teacher_aug_t.log_softmax(dim=-1) * teacher_teacher_aug_t.softmax(dim=-1)).sum(dim=-1).mean()
    return (loss1 + loss2) * 0.5


def batch_images_to_float(batch_images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = batch_images.to(device=device)
    if images.ndim != 4:
        raise ValueError(f"Expected raw images [B,H,W,C] or [B,C,H,W], got {tuple(images.shape)}.")
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2).contiguous()
    if images.shape[1] != 3:
        raise ValueError(f"Expected 3-channel raw images, got {tuple(images.shape)}.")
    return images.float().div(255.0).clamp(0, 1)


def make_rgb_target(batch: dict[str, Any], device: torch.device, output_size: int) -> torch.Tensor:
    if "image" not in batch:
        raise KeyError("RGB low-level training requires dataloader include_raw=True so batch['image'] exists.")
    images = batch_images_to_float(batch["image"], device)
    if images.shape[-1] != output_size or images.shape[-2] != output_size:
        images = F.interpolate(images, size=(output_size, output_size), mode="area")
    return images.clamp(0, 1)


def load_frozen_vae(
    model_root: Path,
    device: torch.device,
    force_float32: bool = False,
) -> AutoencoderKL:
    dtype = torch.float32 if force_float32 or device.type != "cuda" else torch.float16
    vae = AutoencoderKL.from_pretrained(
        model_root / "madebyollin__sdxl-vae-fp16-fix",
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def decode_vae_latents(latents: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    dtype = next(vae.parameters()).dtype
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    decoded = vae.decode(latents.to(dtype=dtype) / scaling_factor).sample
    return (decoded.float() * 0.5 + 0.5).clamp(0, 1)


def image_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return 0.5 * (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy))


def compute_image_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    pred = pred.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    pixel_corr = (
        (pred_centered * target_centered).sum(dim=1)
        / (pred_centered.norm(dim=1) * target_centered.norm(dim=1)).clamp_min(1e-8)
    ).mean()
    mse = F.mse_loss(pred, target)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-8))
    mu_x = pred.mean(dim=(2, 3))
    mu_y = target.mean(dim=(2, 3))
    var_x = pred.var(dim=(2, 3), unbiased=False)
    var_y = target.var(dim=(2, 3), unbiased=False)
    covariance = ((pred - mu_x[:, :, None, None]) * (target - mu_y[:, :, None, None])).mean(dim=(2, 3))
    c1, c2 = 0.01**2, 0.03**2
    ssim = (
        ((2 * mu_x * mu_y + c1) * (2 * covariance + c2))
        / ((mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2))
    ).mean()
    return {"pixcorr": pixel_corr.detach(), "psnr": psnr.detach(), "ssim": ssim.detach()}


def augment_images(images: torch.Tensor) -> torch.Tensor:
    bsz, _, height, width = images.shape
    crop_h = max(1, int(round(height * 0.9)))
    crop_w = max(1, int(round(width * 0.9)))
    augmented = []
    for image in images:
        top = int(torch.randint(0, height - crop_h + 1, (1,), device=images.device).item())
        left = int(torch.randint(0, width - crop_w + 1, (1,), device=images.device).item())
        crop = image[:, top : top + crop_h, left : left + crop_w].unsqueeze(0)
        crop = F.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False).squeeze(0)
        augmented.append(crop)
    out = torch.stack(augmented, dim=0)
    brightness = 1.0 + (torch.rand(bsz, 1, 1, 1, device=images.device) - 0.5) * 0.8
    contrast = 1.0 + (torch.rand(bsz, 1, 1, 1, device=images.device) - 0.5) * 0.8
    mean = out.mean(dim=(2, 3), keepdim=True)
    out = (out - mean) * contrast + mean
    out = out * brightness
    return out.clamp(0, 1)


class FrozenConvNeXtTeacher(nn.Module):
    def __init__(self, variant: str = "tiny", pretrained: bool = True) -> None:
        super().__init__()
        variant = variant.lower()
        if variant == "tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            model = convnext_tiny(weights=weights)
            self.out_dim = 768
        elif variant == "small":
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            model = convnext_small(weights=weights)
            self.out_dim = 768
        elif variant == "base":
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            model = convnext_base(weights=weights)
            self.out_dim = 1024
        else:
            raise ValueError(f"Unsupported ConvNeXt variant: {variant}")
        self.variant = variant
        self.features = model.features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor, output_grid_size: int) -> torch.Tensor:
        images = (images - self.mean) / self.std
        feature_map = self.features(images)
        if feature_map.shape[-1] != output_grid_size or feature_map.shape[-2] != output_grid_size:
            feature_map = F.interpolate(
                feature_map,
                size=(output_grid_size, output_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        return feature_map.flatten(2).permute(0, 2, 1).contiguous()


def compute_low_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    cfg: LowLossConfig,
    convnext_teacher: FrozenConvNeXtTeacher | None,
    vae: AutoencoderKL | None,
) -> dict[str, torch.Tensor]:
    if "vae_latent" in outputs:
        pred = outputs["vae_latent"]
        target = batch["image_vae_latents"].to(device=pred.device, dtype=pred.dtype)
        latent_l1 = F.smooth_l1_loss(pred, target)
        decoded_l1 = pred.new_tensor(0.0)
        decoded_edge = pred.new_tensor(0.0)
        if cfg.decoded_l1_weight > 0 or cfg.decoded_edge_weight > 0:
            if vae is None:
                raise RuntimeError("Decoded image losses require a frozen VAE.")
            n = min(max(cfg.decoded_loss_samples, 1), pred.shape[0])
            decoded = decode_vae_latents(pred[:n], vae)
            image_target = make_rgb_target(batch, pred.device, output_size=decoded.shape[-1])[:n]
            decoded_l1 = F.l1_loss(decoded, image_target)
            decoded_edge = image_edge_loss(decoded, image_target)
        loss_latent_l1 = cfg.latent_l1_weight * latent_l1
        loss_decoded_l1 = cfg.decoded_l1_weight * decoded_l1
        loss_decoded_edge = cfg.decoded_edge_weight * decoded_edge
        loss = loss_latent_l1 + loss_decoded_l1 + loss_decoded_edge
        zero = pred.new_tensor(0.0)
        return {
            "loss": loss,
            "latent_l1": latent_l1.detach(),
            "decoded_l1": decoded_l1.detach(),
            "decoded_edge": decoded_edge.detach(),
            "loss_latent_l1": loss_latent_l1.detach(),
            "loss_decoded_l1": loss_decoded_l1.detach(),
            "loss_decoded_edge": loss_decoded_edge.detach(),
            "low_l1": zero,
            "low_mse": zero,
            "aux_cont": zero,
            "image_mse": zero,
            "image_soft_clip": zero,
            "loss_low_l1": zero,
            "loss_low_mse": zero,
            "loss_aux_cont": zero,
            "loss_image_mse": zero,
            "loss_image_soft_clip": zero,
        }

    pred = outputs["low_level_rgb"]
    target = make_rgb_target(batch, pred.device, output_size=pred.shape[-1]).to(dtype=pred.dtype)
    if tuple(pred.shape) != tuple(target.shape):
        raise ValueError(f"Predicted RGB shape {tuple(pred.shape)} does not match target {tuple(target.shape)}.")
    low_l1 = F.l1_loss(pred, target)
    low_mse = F.mse_loss(pred, target)
    image_mse = pred.new_tensor(0.0)
    image_soft_clip = pred.new_tensor(0.0)
    if convnext_teacher is not None:
        images = batch_images_to_float(batch["image"], pred.device)
        aug_images = augment_images(images)
        grid = int(round(outputs["b_aux"].shape[1] ** 0.5))
        teacher = convnext_teacher(images, output_grid_size=grid)
        teacher_aug = convnext_teacher(aug_images, output_grid_size=grid)
        aux_cont = soft_cont_loss(
            outputs["b_aux"],
            teacher,
            teacher_aug,
            temp=cfg.aux_cont_temp,
            max_tokens=cfg.aux_cont_max_tokens,
        )
    else:
        aux_cont = pred.new_tensor(0.0)

    loss_low_l1 = cfg.low_l1_weight * low_l1
    loss_low_mse = cfg.low_mse_weight * low_mse
    loss_aux_cont = cfg.aux_cont_weight * aux_cont
    loss_image_mse = cfg.image_mse_weight * image_mse
    loss_image_soft_clip = cfg.image_soft_clip_weight * image_soft_clip
    loss = loss_low_l1 + loss_low_mse + loss_aux_cont + loss_image_mse + loss_image_soft_clip
    return {
        "loss": loss,
        "low_l1": low_l1.detach(),
        "low_mse": low_mse.detach(),
        "aux_cont": aux_cont.detach(),
        "image_mse": image_mse.detach(),
        "image_soft_clip": image_soft_clip.detach(),
        "loss_low_l1": loss_low_l1.detach(),
        "loss_low_mse": loss_low_mse.detach(),
        "loss_aux_cont": loss_aux_cont.detach(),
        "loss_image_mse": loss_image_mse.detach(),
        "loss_image_soft_clip": loss_image_soft_clip.detach(),
        "latent_l1": pred.new_tensor(0.0),
        "decoded_l1": pred.new_tensor(0.0),
        "decoded_edge": pred.new_tensor(0.0),
        "loss_latent_l1": pred.new_tensor(0.0),
        "loss_decoded_l1": pred.new_tensor(0.0),
        "loss_decoded_edge": pred.new_tensor(0.0),
    }


def update_meter(total: dict[str, float], values: dict[str, torch.Tensor], n: int) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + float(value.item()) * n


def finalize_meter(total: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in total.items()}


def batch_image_to_pil(batch: dict[str, Any]) -> Image.Image:
    image = batch["image"][0].numpy()
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = np.asarray(image, dtype=np.uint8)
    return Image.fromarray(image, mode="RGB")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().clamp(0, 1)
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def save_original_pred_grid(
    original: Image.Image,
    pred: Image.Image,
    output_path: Path,
    prediction_label: str = "Predicted",
) -> None:
    panel_size = 320
    label_h = 44
    gap = 12
    columns = [("Original", original), (prediction_label, pred)]
    canvas = Image.new("RGB", (panel_size * 2 + gap, panel_size + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in columns:
        canvas.paste(image.resize((panel_size, panel_size), Image.Resampling.BICUBIC), (x, 0))
        draw.text((x + 8, panel_size + 12), label, fill=(0, 0, 0))
        x += panel_size + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def train_one_epoch(
    model: Low1DModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_cfg: LowLossConfig,
    convnext_teacher: FrozenConvNeXtTeacher | None,
    vae: AutoencoderKL | None,
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
        losses = compute_low_loss(outputs, batch, loss_cfg, convnext_teacher, vae)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        n = int(fmri.shape[0])
        update_meter(total, losses, n)
        count += n
    return finalize_meter(total, count)


@torch.no_grad()
def evaluate_model(
    model: Low1DModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_cfg: LowLossConfig,
    convnext_teacher: FrozenConvNeXtTeacher | None,
    vae: AutoencoderKL | None,
    enable_pool: bool,
    pool_num: int,
    pool_type: str,
) -> dict[str, float]:
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
        losses = compute_low_loss(outputs, batch, loss_cfg, convnext_teacher, vae)
        if "low_level_rgb" in outputs:
            predicted_image = outputs["low_level_rgb"]
        else:
            if vae is None:
                raise RuntimeError("VAE evaluation requires a frozen decoder.")
            predicted_image = decode_vae_latents(outputs["vae_latent"], vae)
        image_target = batch_images_to_float(batch["image"], device)
        losses.update(compute_image_metrics(predicted_image, image_target))
        n = int(fmri.shape[0])
        update_meter(total, losses, n)
        count += n
    return finalize_meter(total, count)


@torch.no_grad()
def save_eval_reconstruction(
    model: Low1DModel,
    loader: torch.utils.data.DataLoader,
    output_path: Path,
    device: torch.device,
    enable_pool: bool,
    pool_num: int,
    pool_type: str,
    image_size: int,
    vae: AutoencoderKL | None,
) -> None:
    model.eval()
    batch = next(iter(loader))
    fmri = maybe_pool_fmri(
        batch["fmri"].to(device),
        enable_pool=enable_pool,
        pool_num=pool_num,
        pool_type=pool_type,
    )
    outputs = model(fmri)
    pred_rgb = outputs["low_level_rgb"] if "low_level_rgb" in outputs else decode_vae_latents(outputs["vae_latent"], vae)
    original = batch_image_to_pil(batch)
    pred = tensor_to_pil(pred_rgb[0].cpu())
    if pred.size != (image_size, image_size):
        pred = pred.resize((image_size, image_size), Image.Resampling.BICUBIC)
    label = "Predicted RGB64" if "low_level_rgb" in outputs else "Decoded VAE latent"
    save_original_pred_grid(original, pred, output_path, prediction_label=label)


@torch.no_grad()
def save_fixed_split_reconstructions(
    model: Low1DModel,
    datasets: dict[str, torch.utils.data.Dataset],
    run_dir: Path,
    device: torch.device,
    enable_pool: bool,
    pool_num: int,
    pool_type: str,
    image_size: int,
    sample_count: int,
    vae: AutoencoderKL | None,
) -> dict[str, list[dict[str, Any]]]:
    model.eval()
    manifest: dict[str, list[dict[str, Any]]] = {}
    for split, dataset in datasets.items():
        count = min(max(sample_count, 0), len(dataset))
        indices = np.linspace(0, len(dataset) - 1, num=count, dtype=np.int64).tolist() if count else []
        manifest[split] = []
        for dataset_index in indices:
            item = dataset[int(dataset_index)]
            fmri = maybe_pool_fmri(
                item["fmri"].unsqueeze(0).to(device),
                enable_pool=enable_pool,
                pool_num=pool_num,
                pool_type=pool_type,
            )
            outputs = model(fmri)
            pred_rgb = (
                outputs["low_level_rgb"]
                if "low_level_rgb" in outputs
                else decode_vae_latents(outputs["vae_latent"], vae)
            )[0]
            original_array = item["image"].numpy()
            if original_array.ndim == 3 and original_array.shape[0] == 3:
                original_array = np.transpose(original_array, (1, 2, 0))
            original = Image.fromarray(np.asarray(original_array, dtype=np.uint8), mode="RGB")
            reconstruction = tensor_to_pil(pred_rgb)
            if reconstruction.size != (image_size, image_size):
                reconstruction = reconstruction.resize((image_size, image_size), Image.Resampling.BICUBIC)

            metadata = item["metadata"]
            stem = f"sample_{int(dataset_index):05d}_nsd_{int(metadata['nsd_id']):05d}"
            split_dir = run_dir / "samples" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            comparison_path = split_dir / f"{stem}_comparison.png"
            label = "Predicted RGB64" if "low_level_rgb" in outputs else "Decoded VAE latent"
            save_original_pred_grid(original, reconstruction, comparison_path, prediction_label=label)
            manifest[split].append(
                {
                    "dataset_index": int(dataset_index),
                    "nsd_id": int(metadata["nsd_id"]),
                    "comparison": str(comparison_path.relative_to(run_dir)),
                }
            )
    (run_dir / "samples" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def save_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 1D fMRI model that predicts low-resolution RGB images.")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--stim-info-path", type=Path, default=DEFAULT_STIM_INFO_PATH)
    parser.add_argument("--betas-1d-path", type=Path, default=DEFAULT_BETAS_1D_PATH)
    parser.add_argument("--caption-embeddings-path", type=Path, default=DEFAULT_CAPTION_EMBEDDINGS_PATH)
    parser.add_argument("--image-embeddings-path", type=Path, default=DEFAULT_IMAGE_EMBEDDINGS_PATH)
    parser.add_argument("--image-vae-latents-path", type=Path, default=DEFAULT_IMAGE_VAE_LATENTS_PATH)
    parser.add_argument("--stimulus-h5-path", type=Path, default=DEFAULT_STIMULUS_H5_PATH)
    parser.add_argument("--model-root", type=Path, default=ROOT / "save_pt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output" / "low_model_1D")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--subject", type=int, default=DEFAULT_SUBJECT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--adapter-bottleneck", type=int, default=128)
    parser.add_argument("--spatial-channels", type=int, default=64)
    parser.add_argument("--spatial-seed-size", type=int, default=8)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--aux-dim", type=int, default=768)
    parser.add_argument("--convnext-variant", choices=["tiny", "small", "base"], default="tiny")
    parser.add_argument("--convnext-pretrained", type=str2bool, default=True)
    parser.add_argument("--convnext-loss", type=str2bool, default=False)
    parser.add_argument("--enable-pool", type=str2bool, default=True)
    parser.add_argument("--pool-num", type=int, default=8192)
    parser.add_argument("--pool-type", choices=["max", "avg"], default="max")
    parser.add_argument("--normalize", choices=["none", "volume"], default="volume")
    parser.add_argument("--average-repeats", type=str2bool, default=False)
    parser.add_argument("--voxel-normalize", type=str2bool, default=False)
    parser.add_argument("--target-mode", choices=["rgb", "vae"], default="rgb")
    parser.add_argument("--input-rank", type=int, default=0)
    parser.add_argument("--low-l1-weight", type=float, default=1.0)
    parser.add_argument("--low-mse-weight", type=float, default=0.1)
    parser.add_argument("--aux-cont-weight", type=float, default=0.1)
    parser.add_argument("--image-mse-weight", type=float, default=0.0)
    parser.add_argument("--image-soft-clip-weight", type=float, default=0.0)
    parser.add_argument("--soft-clip-temp", type=float, default=0.005)
    parser.add_argument("--aux-cont-temp", type=float, default=0.2)
    parser.add_argument("--aux-cont-max-tokens", type=int, default=4096)
    parser.add_argument("--latent-l1-weight", type=float, default=1.0)
    parser.add_argument("--decoded-l1-weight", type=float, default=0.0)
    parser.add_argument("--decoded-edge-weight", type=float, default=0.0)
    parser.add_argument("--decoded-loss-samples", type=int, default=8)
    parser.add_argument(
        "--decoded-loss-start-epoch",
        type=int,
        default=1,
        help="First epoch that applies decoded image losses; earlier epochs use latent loss only.",
    )
    parser.add_argument(
        "--best-metric",
        choices=["loss", "pixcorr", "ssim", "visual"],
        default="loss",
        help="Validation criterion used to select the checkpoint. visual averages PixCorr and SSIM.",
    )
    parser.add_argument("--recon-image-size", type=int, default=256)
    parser.add_argument("--save-recon", type=str2bool, default=True)
    parser.add_argument("--fixed-samples-per-split", type=int, default=5)
    parser.add_argument("--delete-checkpoints-after-run", type=str2bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    run_name = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    base_normalize = "none" if args.voxel_normalize else args.normalize
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        annotation_dir=args.annotation_dir,
        stim_info_path=args.stim_info_path,
        betas_1d_path=args.betas_1d_path,
        caption_embeddings_path=args.caption_embeddings_path,
        image_embeddings_path=args.image_embeddings_path,
        image_vae_latents_path=args.image_vae_latents_path,
        stimulus_h5_path=args.stimulus_h5_path,
        fmri_format="1d",
        subject=args.subject,
        seed=args.seed,
        normalize=base_normalize,
        include_vae_latents=args.target_mode == "vae",
        include_raw=True,
    )
    voxel_mean = voxel_std = None
    if args.voxel_normalize or args.average_repeats:
        train_loader, val_loader, test_loader, voxel_mean, voxel_std = make_repeat_averaged_loaders(
            train_loader,
            val_loader,
            test_loader,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    val_recon_loader = DataLoader(
        val_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_nsd_concepts,
    )
    first_batch = next(iter(train_loader))
    raw_in_dim = int(first_batch["fmri"].shape[-1])
    model_in_dim = args.pool_num if args.enable_pool else raw_in_dim
    output_shape = (
        (4, 32, 32)
        if args.target_mode == "vae"
        else (3, int(args.output_size), int(args.output_size))
    )
    image_embedding_dim = int(first_batch["image_embeddings"].shape[-1])

    resume_checkpoint: dict[str, Any] | None = None
    resume_path = Path(args.resume_checkpoint).expanduser() if str(args.resume_checkpoint).strip() else None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Missing resume checkpoint: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location="cpu")

    if isinstance(resume_checkpoint, dict) and "model_config" in resume_checkpoint:
        model_cfg = dataclass_from_payload(Low1DConfig, resume_checkpoint["model_config"])
        if model_cfg.in_dim != model_in_dim:
            raise ValueError(
                f"Resume checkpoint expects in_dim={model_cfg.in_dim}, but current data/model input is {model_in_dim}."
            )
        if model_cfg.image_embedding_dim != image_embedding_dim:
            raise ValueError(
                f"Resume checkpoint expects image_embedding_dim={model_cfg.image_embedding_dim}, "
                f"but current labels use {image_embedding_dim}."
            )
        if model_cfg.target_mode != args.target_mode:
            raise ValueError(
                f"Resume checkpoint target_mode={model_cfg.target_mode}, current={args.target_mode}."
            )
        if args.target_mode == "rgb" and (
            model_cfg.output_channels != output_shape[0] or model_cfg.output_size != output_shape[1]
        ):
            raise ValueError(
                "Resume checkpoint output shape does not match current RGB targets: "
                f"checkpoint=({model_cfg.output_channels}, {model_cfg.output_size}, {model_cfg.output_size}), "
                f"current={output_shape}."
            )
    else:
        model_cfg = Low1DConfig(
            in_dim=model_in_dim,
            hidden_dim=args.hidden_dim,
            n_blocks=args.n_blocks,
            adapter_bottleneck=args.adapter_bottleneck,
            dropout=args.dropout,
            image_embedding_dim=image_embedding_dim,
            spatial_channels=args.spatial_channels,
            spatial_seed_size=args.spatial_seed_size,
            aux_dim=args.aux_dim,
            output_channels=output_shape[0],
            output_size=output_shape[1],
            target_mode=args.target_mode,
            input_rank=args.input_rank,
            latent_channels=4,
            latent_size=32,
        )
    loss_cfg = LowLossConfig(
        low_l1_weight=args.low_l1_weight,
        low_mse_weight=args.low_mse_weight,
        aux_cont_weight=args.aux_cont_weight,
        image_mse_weight=args.image_mse_weight,
        image_soft_clip_weight=args.image_soft_clip_weight,
        soft_clip_temp=args.soft_clip_temp,
        aux_cont_temp=args.aux_cont_temp,
        aux_cont_max_tokens=args.aux_cont_max_tokens,
        latent_l1_weight=args.latent_l1_weight,
        decoded_l1_weight=args.decoded_l1_weight,
        decoded_edge_weight=args.decoded_edge_weight,
        decoded_loss_samples=args.decoded_loss_samples,
    )
    model = Low1DModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resume_start_epoch = 0
    if resume_checkpoint is not None:
        if isinstance(resume_checkpoint, dict) and "model_state_dict" in resume_checkpoint:
            model.load_state_dict(resume_checkpoint["model_state_dict"])
            if "optimizer_state_dict" in resume_checkpoint:
                optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
                move_optimizer_state_to_device(optimizer, device)
            resume_start_epoch = int(resume_checkpoint.get("epoch", 0))
        elif isinstance(resume_checkpoint, dict):
            model.load_state_dict(resume_checkpoint)
        else:
            raise TypeError(f"Unsupported checkpoint payload type: {type(resume_checkpoint)!r}")

    config_payload = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": asdict(model_cfg),
        "loss": asdict(loss_cfg),
        "raw_in_dim": raw_in_dim,
        "model_in_dim": model_in_dim,
        "output_shape": output_shape,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "resume_checkpoint": None if resume_path is None else str(resume_path),
        "resume_start_epoch": resume_start_epoch,
        "train_additional_epochs": args.epochs,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
    }
    if voxel_mean is not None and voxel_std is not None:
        np.savez_compressed(
            run_dir / "voxel_stats.npz",
            mean=voxel_mean.numpy(),
            std=voxel_std.numpy(),
        )
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)
    print(f"run_dir={run_dir}")
    print(f"model params={config_payload['parameters']:,} trainable={config_payload['trainable_parameters']:,}")
    print(f"raw_in_dim={raw_in_dim} model_in_dim={model_in_dim} output_shape={output_shape}")
    if resume_path is not None:
        print(f"resumed from {resume_path} at epoch={resume_start_epoch}; training {args.epochs} more epochs")

    convnext_teacher = None
    if args.convnext_loss:
        convnext_teacher = FrozenConvNeXtTeacher(
            variant=args.convnext_variant,
            pretrained=args.convnext_pretrained,
        ).to(device)
        if model_cfg.aux_dim != convnext_teacher.out_dim:
            raise ValueError(
                f"Model aux_dim={model_cfg.aux_dim} must match ConvNeXt {args.convnext_variant} "
                f"feature dim={convnext_teacher.out_dim}. Set --aux-dim {convnext_teacher.out_dim}."
            )

    vae = None
    if args.target_mode == "vae":
        vae = load_frozen_vae(
            args.model_root,
            device,
            force_float32=args.decoded_l1_weight > 0 or args.decoded_edge_weight > 0,
        )

    best_loss = float("inf")
    best_score = float("inf") if args.best_metric == "loss" else -float("inf")
    best_epoch = -1
    final_epoch = resume_start_epoch + args.epochs
    for epoch in range(resume_start_epoch + 1, final_epoch + 1):
        decoded_loss_active = epoch >= args.decoded_loss_start_epoch
        epoch_loss_cfg = loss_cfg
        if not decoded_loss_active:
            epoch_loss_cfg = replace(loss_cfg, decoded_l1_weight=0.0, decoded_edge_weight=0.0)
        start = time.perf_counter()
        train_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch_loss_cfg,
            convnext_teacher,
            vae,
            enable_pool=args.enable_pool,
            pool_num=args.pool_num,
            pool_type=args.pool_type,
        )
        train_time = time.perf_counter() - start
        payload: dict[str, Any] = {"epoch": epoch, "train": train_losses, "train_time_sec": train_time}
        message = (
            f"epoch={epoch:03d} train_loss={train_losses['loss']:.6f} "
            f"low_l1*w={train_losses['loss_low_l1']:.6f} "
            f"low_mse*w={train_losses['loss_low_mse']:.6f} "
            f"aux_cont*w={train_losses['loss_aux_cont']:.6f} "
            f"img_mse*w={train_losses['loss_image_mse']:.6f} "
            f"img_sc*w={train_losses['loss_image_soft_clip']:.6f} "
            f"latent_l1*w={train_losses['loss_latent_l1']:.6f} "
            f"decoded_l1*w={train_losses['loss_decoded_l1']:.6f} "
            f"edge*w={train_losses['loss_decoded_edge']:.6f} "
            f"time={train_time:.1f}s"
        )

        do_eval = args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == final_epoch)
        if do_eval:
            eval_start = time.perf_counter()
            val_losses = evaluate_model(
                model,
                val_loader,
                device,
                epoch_loss_cfg,
                convnext_teacher,
                vae,
                enable_pool=args.enable_pool,
                pool_num=args.pool_num,
                pool_type=args.pool_type,
            )
            eval_time = time.perf_counter() - eval_start
            payload["val"] = val_losses
            payload["val_time_sec"] = eval_time
            best_loss = min(best_loss, val_losses["loss"])
            if args.best_metric == "loss":
                current_score = val_losses["loss"]
                is_best = current_score < best_score
            elif args.best_metric == "visual":
                current_score = 0.5 * (val_losses["pixcorr"] + val_losses["ssim"])
                is_best = current_score > best_score
            else:
                current_score = val_losses[args.best_metric]
                is_best = current_score > best_score
            if is_best:
                best_score = current_score
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_config": asdict(model_cfg),
                        "epoch": epoch,
                        "best_loss": best_loss,
                        "best_metric": args.best_metric,
                        "best_score": best_score,
                        "best_epoch": best_epoch,
                    },
                    run_dir / "best_low_model.pt",
                )
            message += (
                f" val_loss={val_losses['loss']:.6f}"
                f" val_low_l1*w={val_losses['loss_low_l1']:.6f}"
                f" val_low_mse*w={val_losses['loss_low_mse']:.6f}"
                f" val_aux_cont*w={val_losses['loss_aux_cont']:.6f}"
                f" val_img_mse*w={val_losses['loss_image_mse']:.6f}"
                f" val_img_sc*w={val_losses['loss_image_soft_clip']:.6f}"
                f" val_latent_l1*w={val_losses['loss_latent_l1']:.6f}"
                f" val_decoded_l1*w={val_losses['loss_decoded_l1']:.6f}"
                f" val_edge*w={val_losses['loss_decoded_edge']:.6f}"
                f" pixcorr={val_losses['pixcorr']:.4f}"
                f" ssim={val_losses['ssim']:.4f}"
                f" psnr={val_losses['psnr']:.2f}"
                f" best_{args.best_metric}={best_score:.6f}@{best_epoch}"
                f" eval_time={eval_time:.1f}s"
            )
            if args.save_recon:
                recon_path = run_dir / f"recon_val_{epoch:03d}.png"
                save_eval_reconstruction(
                    model,
                    val_recon_loader,
                    recon_path,
                    device,
                    enable_pool=args.enable_pool,
                    pool_num=args.pool_num,
                    pool_type=args.pool_type,
                    image_size=args.recon_image_size,
                    vae=vae,
                )
                payload["recon_path"] = str(recon_path)
                message += f" recon={recon_path.name}"
        print(message)
        save_jsonl(metrics_path, payload)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_cfg),
            "epoch": final_epoch,
            "best_loss": best_loss,
            "best_metric": args.best_metric,
            "best_score": best_score,
            "best_epoch": best_epoch,
        },
        run_dir / "last_low_model.pt",
    )
    best_path = run_dir / "best_low_model.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    test_losses = evaluate_model(
        model,
        test_loader,
        device,
        loss_cfg,
        convnext_teacher,
        vae,
        enable_pool=args.enable_pool,
        pool_num=args.pool_num,
        pool_type=args.pool_type,
    )
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_losses, f, ensure_ascii=False, indent=2)
    if args.save_recon:
        manifest = save_fixed_split_reconstructions(
            model,
            {"train": train_loader.dataset, "val": val_loader.dataset, "test": test_loader.dataset},
            run_dir,
            device,
            enable_pool=args.enable_pool,
            pool_num=args.pool_num,
            pool_type=args.pool_type,
            image_size=args.recon_image_size,
            sample_count=args.fixed_samples_per_split,
            vae=vae,
        )
        print(
            "saved fixed reconstructions "
            + " ".join(f"{split}={len(samples)}" for split, samples in manifest.items())
        )
    print(
        f"test_loss={test_losses['loss']:.6f} "
        f"test_low_l1={test_losses['low_l1']:.6f} "
        f"test_low_mse={test_losses['low_mse']:.6f} "
        f"test_aux_cont={test_losses['aux_cont']:.6f} "
        f"test_image_mse={test_losses['image_mse']:.6f} "
        f"test_image_soft_clip={test_losses['image_soft_clip']:.6f}"
        f" test_latent_l1={test_losses['latent_l1']:.6f}"
        f" test_decoded_l1={test_losses['decoded_l1']:.6f}"
        f" test_decoded_edge={test_losses['decoded_edge']:.6f}"
    )
    if args.delete_checkpoints_after_run:
        deleted = []
        for checkpoint_path in run_dir.glob("*.pt"):
            checkpoint_path.unlink()
            deleted.append(checkpoint_path.name)
        print(f"deleted checkpoints after evaluation: {deleted}")
    print(f"saved outputs to {run_dir}")


if __name__ == "__main__":
    main()
