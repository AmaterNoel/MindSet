from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import open_clip
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NSD_ROOT = Path(r"D:\datasets\NSD")
DEFAULT_MODEL_ROOT = ROOT / "save_pt"
DEFAULT_MAX_CAPTIONS = 5
COCO_SPLITS = ("train2017", "val2017")


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_subject(value: str) -> tuple[str, int, str]:
    raw = str(value).strip()
    if raw.lower().startswith("subj"):
        number = int(raw[4:])
    elif raw.lower().startswith("s"):
        number = int(raw[1:])
    else:
        number = int(raw)
    return f"S{number}", number, f"subj{number:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OpenCLIP ViT-bigG-14 text caption embeddings and image embeddings for NSD subject stimuli."
    )
    parser.add_argument("--subject", default="S1", help="Subject id, e.g. S1, S2, 1, or subj01.")
    parser.add_argument("--nsd-root", type=Path, default=DEFAULT_NSD_ROOT)
    parser.add_argument("--annotation-dir", type=Path, default=None)
    parser.add_argument("--stimulus-h5", type=Path, default=None)
    parser.add_argument("--stim-info", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--caption-output", type=Path, default=None)
    parser.add_argument("--image-output", type=Path, default=None)
    parser.add_argument("--vae-output", type=Path, default=None)
    parser.add_argument("--vae-image-size", type=int, default=256)
    parser.add_argument("--max-captions", type=int, default=DEFAULT_MAX_CAPTIONS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--overwrite", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--reuse-existing", type=str2bool, nargs="?", const=True, default=True)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    subject_name, subject_number, subject_dir_name = parse_subject(args.subject)
    args.subject_name = subject_name
    args.subject_number = subject_number
    args.subject_dir_name = subject_dir_name

    args.annotation_dir = args.annotation_dir or args.nsd_root / "annotations"
    args.stim_info = args.stim_info or args.annotation_dir / "nsd_stim_info_merged.csv"
    args.stimulus_h5 = args.stimulus_h5 or args.nsd_root / "stimulus" / f"{subject_name}_stimuli_label_order_256.h5py"
    args.output_dir = args.output_dir or args.nsd_root / subject_dir_name
    args.caption_output = args.caption_output or args.output_dir / "caption_text_embeddings.pt"
    args.image_output = args.image_output or args.output_dir / "image_embeddings.pt"
    args.vae_output = args.vae_output or args.output_dir / "image_vae_latents.pt"
    return args


def load_captions_by_split(annotation_dir: Path, max_captions: int) -> dict[str, dict[int, list[str]]]:
    captions_by_split: dict[str, dict[int, list[str]]] = {}
    for split in COCO_SPLITS:
        path = annotation_dir / f"captions_{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing COCO captions file: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        captions_by_image: dict[int, list[str]] = defaultdict(list)
        for ann in data.get("annotations", []):
            image_id = int(ann["image_id"])
            caption = str(ann["caption"]).strip()
            if caption and len(captions_by_image[image_id]) < max_captions:
                captions_by_image[image_id].append(caption)
        captions_by_split[split] = dict(captions_by_image)
    return captions_by_split


def pad_captions(captions: list[str], max_captions: int) -> tuple[list[str], list[bool]]:
    clipped = captions[:max_captions]
    mask = [True] * len(clipped)
    while len(clipped) < max_captions:
        clipped.append("")
        mask.append(False)
    return clipped, mask


def load_subject_records(
    stim_info: Path,
    annotation_dir: Path,
    subject_number: int,
    max_captions: int,
) -> tuple[list[dict[str, Any]], list[list[str]], torch.Tensor]:
    if not stim_info.exists():
        raise FileNotFoundError(f"Missing NSD stimulus info CSV: {stim_info}")

    subject_col = f"subject{subject_number}"
    with stim_info.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required = {"nsdId", "cocoId", "cocoSplit", subject_col}
    missing = required - fieldnames
    if missing:
        raise KeyError(f"Stimulus CSV is missing columns: {sorted(missing)}")

    captions_by_split = load_captions_by_split(annotation_dir, max_captions)
    ordered_rows = [
        row
        for split in COCO_SPLITS
        for row in rows
        if row["cocoSplit"] == split and int(row[subject_col]) == 1
    ]

    records: list[dict[str, Any]] = []
    all_captions: list[list[str]] = []
    caption_masks: list[list[bool]] = []
    for h5_index, row in enumerate(ordered_rows):
        coco_split = str(row["cocoSplit"])
        coco_id = int(row["cocoId"])
        captions, caption_mask = pad_captions(
            captions_by_split.get(coco_split, {}).get(coco_id, []),
            max_captions,
        )
        records.append(
            {
                "h5_index": h5_index,
                "nsd_id": int(row["nsdId"]),
                "coco_id": coco_id,
                "coco_split": coco_split,
            }
        )
        all_captions.append(captions)
        caption_masks.append(caption_mask)

    return records, all_captions, torch.tensor(caption_masks, dtype=torch.bool)


class H5StimulusDataset(Dataset):
    def __init__(self, path: Path, preprocess: Any) -> None:
        self.path = path
        self.preprocess = preprocess
        self._file: h5py.File | None = None
        with h5py.File(self.path, "r") as f:
            if "stimuli" not in f:
                raise KeyError(f"{self.path} does not contain dataset 'stimuli'")
            self.length = int(f["stimuli"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        image = self._file["stimuli"][index]
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return self.preprocess(Image.fromarray(image).convert("RGB"))


class H5VAEStimulusDataset(Dataset):
    def __init__(self, path: Path, image_size: int) -> None:
        if image_size <= 0 or image_size % 8 != 0:
            raise ValueError("vae_image_size must be a positive multiple of 8.")
        self.path = path
        self.image_size = image_size
        self._file: h5py.File | None = None
        with h5py.File(self.path, "r") as f:
            if "stimuli" not in f:
                raise KeyError(f"{self.path} does not contain dataset 'stimuli'")
            self.length = int(f["stimuli"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        image = np.asarray(self._file["stimuli"][index])
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image).convert("RGB")
        if pil_image.size != (self.image_size, self.image_size):
            pil_image = pil_image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(pil_image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_openclip_model(args: argparse.Namespace, dtype: torch.dtype):
    model_dir = args.model_root / "laion__CLIP-ViT-bigG-14-laion2B-39B-b160k"
    weights = model_dir / "open_clip_model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"Missing OpenCLIP weights: {weights}")

    precision = "fp16" if dtype == torch.float16 and str(args.device).startswith("cuda") else "fp32"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-bigG-14",
        pretrained=str(weights),
        device=args.device,
        precision=precision,
    )
    tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
    model.eval()
    return model, tokenizer, preprocess


def load_sdxl_vae(args: argparse.Namespace, dtype: torch.dtype) -> AutoencoderKL:
    vae_dir = args.model_root / "madebyollin__sdxl-vae-fp16-fix"
    if not vae_dir.exists():
        raise FileNotFoundError(f"Missing SDXL VAE directory: {vae_dir}")
    model_dtype = dtype if dtype == torch.float16 and str(args.device).startswith("cuda") else torch.float32
    vae = AutoencoderKL.from_pretrained(
        vae_dir,
        torch_dtype=model_dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    vae.to(args.device)
    vae.eval()
    return vae


def get_visual_input_dtype(model: torch.nn.Module) -> torch.dtype:
    conv1 = getattr(model.visual, "conv1", None)
    if conv1 is not None and hasattr(conv1, "weight"):
        return conv1.weight.dtype
    return next(model.visual.parameters()).dtype


@torch.no_grad()
def encode_caption_texts(
    model: torch.nn.Module,
    tokenizer: Any,
    captions: list[list[str]],
    caption_mask: torch.Tensor,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    flat_captions = [caption for item in captions for caption in item]
    embeddings: list[torch.Tensor] = []
    for start in range(0, len(flat_captions), batch_size):
        batch = flat_captions[start : start + batch_size]
        tokens = tokenizer(batch).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        embeddings.append(emb.detach().cpu().to(dtype=dtype))
        if start == 0 or (start // batch_size + 1) % 50 == 0:
            print(f"encoded captions: {min(start + len(batch), len(flat_captions))}/{len(flat_captions)}")

    all_embeddings = torch.cat(embeddings, dim=0)
    all_embeddings = all_embeddings.view(len(captions), len(captions[0]), -1)
    all_embeddings[~caption_mask] = 0
    return all_embeddings.contiguous()


@torch.no_grad()
def encode_images(
    model: torch.nn.Module,
    preprocess: Any,
    stimulus_h5: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    dataset = H5StimulusDataset(stimulus_h5, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )
    embeddings: list[torch.Tensor] = []
    visual_dtype = get_visual_input_dtype(model)
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device=device, dtype=visual_dtype, non_blocking=True)
        if batch_idx == 0:
            print(f"image encoder dtype: input={batch.dtype}, visual={visual_dtype}")
        emb = model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        embeddings.append(emb.detach().cpu().to(dtype=dtype))
        if batch_idx == 0 or (batch_idx + 1) % 20 == 0:
            done = min((batch_idx + 1) * batch_size, len(dataset))
            print(f"encoded images: {done}/{len(dataset)}")
    return torch.cat(embeddings, dim=0).contiguous()


@torch.no_grad()
def encode_vae_latents(
    vae: AutoencoderKL,
    stimulus_h5: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    dataset = H5VAEStimulusDataset(stimulus_h5, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )
    latents: list[torch.Tensor] = []
    vae_dtype = next(vae.parameters()).dtype
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device=device, dtype=vae_dtype, non_blocking=True)
        batch = batch * 2.0 - 1.0
        if batch_idx == 0:
            print(f"vae encoder dtype: input={batch.dtype}, vae={vae_dtype}")
        latent = vae.encode(batch).latent_dist.mode()
        latent = latent * scaling_factor
        latents.append(latent.detach().cpu().to(dtype=dtype))
        if batch_idx == 0 or (batch_idx + 1) % 20 == 0:
            done = min((batch_idx + 1) * batch_size, len(dataset))
            print(f"encoded vae latents: {done}/{len(dataset)}")
    return torch.cat(latents, dim=0).contiguous(), scaling_factor


def ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def save_atomic(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def main() -> None:
    args = resolve_paths(parse_args())
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    need_caption = not (args.reuse_existing and args.caption_output.exists())
    need_image = not (args.reuse_existing and args.image_output.exists())
    need_vae = not (args.reuse_existing and args.vae_output.exists())
    if need_caption:
        ensure_can_write(args.caption_output, args.overwrite)
    if need_image:
        ensure_can_write(args.image_output, args.overwrite)
    if need_vae:
        ensure_can_write(args.vae_output, args.overwrite)

    records, captions, caption_mask = load_subject_records(
        stim_info=args.stim_info,
        annotation_dir=args.annotation_dir,
        subject_number=args.subject_number,
        max_captions=args.max_captions,
    )
    with h5py.File(args.stimulus_h5, "r") as f:
        if "stimuli" not in f:
            raise KeyError(f"{args.stimulus_h5} does not contain dataset 'stimuli'")
        stimulus_count = int(f["stimuli"].shape[0])
    if stimulus_count != len(records):
        raise RuntimeError(
            f"Stimulus count mismatch: {args.stimulus_h5} has {stimulus_count} images, "
            f"but annotations map {len(records)} rows."
        )

    print(f"subject: {args.subject_name}")
    print(f"stimuli: {stimulus_count}")
    print(f"caption output: {args.caption_output}")
    print(f"image output: {args.image_output}")
    print(f"vae output: {args.vae_output}")

    if need_caption or need_image:
        model, tokenizer, preprocess = load_openclip_model(args, dtype)
    else:
        model = tokenizer = preprocess = None

    if need_caption:
        assert model is not None and tokenizer is not None
        caption_embeddings = encode_caption_texts(
            model=model,
            tokenizer=tokenizer,
            captions=captions,
            caption_mask=caption_mask,
            batch_size=args.batch_size,
            device=args.device,
            dtype=dtype,
        )
        save_atomic(
            {
                "subject": args.subject_name,
                "model": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                "embedding_dim": int(caption_embeddings.shape[-1]),
                "captions": captions,
                "caption_mask": caption_mask,
                "caption_text_embeddings": caption_embeddings,
                "records": records,
            },
            args.caption_output,
        )
        print(f"caption_text_embeddings: {tuple(caption_embeddings.shape)} {caption_embeddings.dtype}")
    else:
        print(f"skip existing caption embeddings: {args.caption_output}")

    if need_image:
        assert model is not None and preprocess is not None
        image_embeddings = encode_images(
            model=model,
            preprocess=preprocess,
            stimulus_h5=args.stimulus_h5,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            dtype=dtype,
        )
        save_atomic(
            {
                "subject": args.subject_name,
                "model": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                "embedding_dim": int(image_embeddings.shape[-1]),
                "image_embeddings": image_embeddings,
                "records": records,
            },
            args.image_output,
        )
        print(f"image_embeddings: {tuple(image_embeddings.shape)} {image_embeddings.dtype}")
    else:
        print(f"skip existing image embeddings: {args.image_output}")

    if need_vae:
        vae = load_sdxl_vae(args, dtype)
        image_vae_latents, scaling_factor = encode_vae_latents(
            vae=vae,
            stimulus_h5=args.stimulus_h5,
            image_size=args.vae_image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            dtype=dtype,
        )
        save_atomic(
            {
                "subject": args.subject_name,
                "model": "madebyollin/sdxl-vae-fp16-fix",
                "image_size": int(args.vae_image_size),
                "latent_scaling_factor": scaling_factor,
                "image_vae_latents": image_vae_latents,
                "records": records,
            },
            args.vae_output,
        )
        print(f"image_vae_latents: {tuple(image_vae_latents.shape)} {image_vae_latents.dtype}")
    else:
        print(f"skip existing vae latents: {args.vae_output}")

    if not need_caption and not need_image and not need_vae:
        print("all requested embedding files already exist.")


if __name__ == "__main__":
    main()
