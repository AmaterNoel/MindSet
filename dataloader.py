from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_NSD_ROOT = Path(r"D:\datasets\NSD")
DEFAULT_ANNOTATION_DIR = DEFAULT_NSD_ROOT / "annotations"
DEFAULT_STIM_INFO_PATH = DEFAULT_ANNOTATION_DIR / "nsd_stim_info_merged.csv"
DEFAULT_BETAS_PATH = DEFAULT_NSD_ROOT / "subj01" / "betas_float16.npy"
DEFAULT_CAPTION_EMBEDDINGS_PATH = DEFAULT_NSD_ROOT / "subj01" / "caption_text_embeddings.pt"
DEFAULT_IMAGE_EMBEDDINGS_PATH = DEFAULT_NSD_ROOT / "subj01" / "image_embeddings.pt"
DEFAULT_IMAGE_VAE_LATENTS_PATH = DEFAULT_NSD_ROOT / "subj01" / "image_vae_latents.pt"
DEFAULT_STIMULUS_H5_PATH = DEFAULT_NSD_ROOT / "stimulus" / "S1_stimuli_label_order_256.h5py"

DEFAULT_SUBJECT = 1
DEFAULT_SESSIONS = 40
DEFAULT_TRIALS_PER_SESSION = 750
DEFAULT_MAX_CAPTIONS = 5
DEFAULT_TRAIN_RATIO = 0.9
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SPLIT_SEED = 42
COCO_SPLITS = ("train2017", "val2017")

SplitName = Literal["train", "val", "test", "all"]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def load_coco_captions(annotation_dir: Path, max_captions: int) -> dict[tuple[str, int], list[str]]:
    captions_by_image: dict[tuple[str, int], list[str]] = defaultdict(list)
    for split in COCO_SPLITS:
        path = annotation_dir / f"captions_{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing COCO captions file: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for ann in data.get("annotations", []):
            image_id = int(ann["image_id"])
            caption = str(ann["caption"]).strip()
            key = (split, image_id)
            if caption and len(captions_by_image[key]) < max_captions:
                captions_by_image[key].append(caption)
    return dict(captions_by_image)


def pad_captions(captions: list[str], max_captions: int) -> tuple[list[str], list[bool]]:
    clipped = captions[:max_captions]
    mask = [True] * len(clipped)
    while len(clipped) < max_captions:
        clipped.append("")
        mask.append(False)
    return clipped, mask


def build_trial_records(
    stim_info_path: Path,
    annotation_dir: Path,
    subject: int,
    sessions: int,
    trials_per_session: int,
    max_captions: int,
) -> list[dict[str, Any]]:
    if not stim_info_path.exists():
        raise FileNotFoundError(f"Missing NSD stimulus index: {stim_info_path}")

    subject_col = f"subject{subject}"
    rep_columns = [f"subject{subject}_rep{i}" for i in range(3)]
    max_global_trial = sessions * trials_per_session
    captions_by_image = load_coco_captions(annotation_dir, max_captions)

    with stim_info_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    missing = {"nsdId", "cocoId", "cocoSplit", subject_col, *rep_columns} - fieldnames
    if missing:
        raise KeyError(f"Stimulus CSV is missing columns: {sorted(missing)}")

    ordered_rows = [
        row
        for split in COCO_SPLITS
        for row in rows
        if row["cocoSplit"] == split and int(row[subject_col]) == 1
    ]

    records: list[dict[str, Any]] = []
    for label_index, row in enumerate(ordered_rows):
        coco_split = str(row["cocoSplit"])
        coco_id = int(row["cocoId"])
        captions = captions_by_image.get((coco_split, coco_id), [])
        if not captions:
            raise KeyError(f"Missing captions for {coco_split}/{coco_id}")
        padded_captions, caption_mask = pad_captions(captions, max_captions)

        for rep_idx, column in enumerate(rep_columns):
            global_trial = int(row[column])
            if global_trial <= 0 or global_trial > max_global_trial:
                continue
            beta_index = (global_trial - 1) % trials_per_session
            records.append(
                {
                    "label_index": label_index,
                    "global_trial": global_trial,
                    "beta_row": global_trial - 1,
                    "session": (global_trial - 1) // trials_per_session + 1,
                    "trial_in_session": beta_index + 1,
                    "beta_index": beta_index,
                    "rep": rep_idx,
                    "nsd_id": int(row["nsdId"]),
                    "coco_id": coco_id,
                    "coco_split": coco_split,
                    "captions": padded_captions,
                    "caption_mask": caption_mask,
                }
            )

    records.sort(key=lambda item: int(item["global_trial"]))
    return records


def split_group_ids(
    group_ids: list[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= val_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1), val_ratio must be in [0,1).")
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio must equal 1.0 for train2017 train/val split.")

    groups = sorted(set(int(group_id) for group_id in group_ids))
    rng = np.random.default_rng(seed)
    shuffled = [groups[i] for i in rng.permutation(len(groups))]
    n_train = int(round(len(shuffled) * train_ratio))
    return set(shuffled[:n_train]), set(shuffled[n_train:])


def deterministic_group_split(
    records: list[dict[str, Any]],
    split: SplitName,
    group_key: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> list[int]:
    if split == "all":
        return list(range(len(records)))
    if not records:
        return []
    if group_key not in records[0]:
        raise KeyError(f"group_key {group_key!r} is not present in records.")

    train_records = [record for record in records if str(record["coco_split"]) == "train2017"]
    test_indices = [idx for idx, record in enumerate(records) if str(record["coco_split"]) == "val2017"]
    train_groups, val_groups = split_group_ids(
        [int(record[group_key]) for record in train_records],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    if split == "test":
        return test_indices
    selected = train_groups if split == "train" else val_groups
    return [idx for idx, record in enumerate(records) if int(record[group_key]) in selected]


def check_no_group_leakage(
    records: list[dict[str, Any]],
    group_key: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    train_records = [record for record in records if str(record["coco_split"]) == "train2017"]
    test_records = [record for record in records if str(record["coco_split"]) == "val2017"]
    train_groups, val_groups = split_group_ids(
        [int(record[group_key]) for record in train_records],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    test_groups = {int(record[group_key]) for record in test_records}
    overlaps = [
        ("train", "val", train_groups & val_groups),
        ("train", "test", train_groups & test_groups),
        ("val", "test", val_groups & test_groups),
    ]
    leaked = [(left, right, sorted(values)[:10]) for left, right, values in overlaps if values]
    if leaked:
        raise RuntimeError(f"Group leakage detected for {group_key}: {leaked}")


def normalize_volume(volume: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return volume
    if mode != "volume":
        raise ValueError(f"Unknown normalize mode: {mode}")
    valid = np.isfinite(volume)
    values = volume[valid]
    if values.size == 0:
        return volume
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        std = 1.0
    return (volume - mean) / std


def load_label_tensor(path: Path, key: str) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing label embedding file: {path}")
    payload = torch.load(path, map_location="cpu")
    if key not in payload:
        raise KeyError(f"{path} is missing key {key!r}")
    tensor = payload[key]
    if not torch.is_tensor(tensor):
        raise TypeError(f"{path}:{key} is not a tensor.")
    return tensor


def load_caption_mask(path: Path, fallback_shape: tuple[int, int]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    mask = payload.get("caption_mask")
    if mask is None:
        return torch.ones(fallback_shape, dtype=torch.bool)
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    return mask.to(dtype=torch.bool)


class NSDConceptDataset(Dataset):
    def __init__(
        self,
        annotation_dir: Path = DEFAULT_ANNOTATION_DIR,
        stim_info_path: Path = DEFAULT_STIM_INFO_PATH,
        betas_path: Path = DEFAULT_BETAS_PATH,
        caption_embeddings_path: Path = DEFAULT_CAPTION_EMBEDDINGS_PATH,
        image_embeddings_path: Path = DEFAULT_IMAGE_EMBEDDINGS_PATH,
        image_vae_latents_path: Path = DEFAULT_IMAGE_VAE_LATENTS_PATH,
        stimulus_h5_path: Path = DEFAULT_STIMULUS_H5_PATH,
        split: SplitName = "train",
        group_key: str = "nsd_id",
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        val_ratio: float = DEFAULT_VAL_RATIO,
        seed: int = DEFAULT_SPLIT_SEED,
        subject: int = DEFAULT_SUBJECT,
        sessions: int = DEFAULT_SESSIONS,
        trials_per_session: int = DEFAULT_TRIALS_PER_SESSION,
        max_captions: int = DEFAULT_MAX_CAPTIONS,
        normalize: str = "volume",
        include_vae_latents: bool = True,
        include_raw: bool = False,
        records: list[dict[str, Any]] | None = None,
    ) -> None:
        self.annotation_dir = Path(annotation_dir)
        self.stim_info_path = Path(stim_info_path)
        self.betas_path = Path(betas_path)
        self.caption_embeddings_path = Path(caption_embeddings_path)
        self.image_embeddings_path = Path(image_embeddings_path)
        self.image_vae_latents_path = Path(image_vae_latents_path)
        self.stimulus_h5_path = Path(stimulus_h5_path)
        self.split = split
        self.group_key = group_key
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.subject = subject
        self.sessions = sessions
        self.trials_per_session = trials_per_session
        self.max_captions = max_captions
        self.normalize = normalize
        self.include_vae_latents = include_vae_latents
        self.include_raw = include_raw
        self._stimulus_h5: h5py.File | None = None

        self.records = records or build_trial_records(
            stim_info_path=self.stim_info_path,
            annotation_dir=self.annotation_dir,
            subject=self.subject,
            sessions=self.sessions,
            trials_per_session=self.trials_per_session,
            max_captions=self.max_captions,
        )
        if not self.records:
            raise RuntimeError("No NSD trials matched the selected subject.")
        if self.group_key != "nsd_id":
            raise ValueError("Use group_key='nsd_id' to avoid leakage across repeated NSD trials.")

        check_no_group_leakage(
            self.records,
            group_key=self.group_key,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
        self.indices = deterministic_group_split(
            self.records,
            split=self.split,
            group_key=self.group_key,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )

        if not self.betas_path.exists():
            raise FileNotFoundError(f"Missing beta cache: {self.betas_path}")
        self.betas = np.load(self.betas_path, mmap_mode="r")
        if self.betas.ndim != 4:
            raise ValueError(f"Expected beta cache shape [trials, x, y, z], got {self.betas.shape}.")
        max_beta_row = max(int(record["beta_row"]) for record in self.records)
        if max_beta_row >= self.betas.shape[0]:
            raise ValueError(f"Beta cache has {self.betas.shape[0]} rows but needs row {max_beta_row}.")

        self.caption_text_embeddings = load_label_tensor(
            self.caption_embeddings_path,
            "caption_text_embeddings",
        )
        self.image_embeddings = load_label_tensor(self.image_embeddings_path, "image_embeddings")
        self.image_vae_latents = None
        if self.include_vae_latents:
            self.image_vae_latents = load_label_tensor(self.image_vae_latents_path, "image_vae_latents")
        self.caption_mask = load_caption_mask(
            self.caption_embeddings_path,
            fallback_shape=(self.caption_text_embeddings.shape[0], self.caption_text_embeddings.shape[1]),
        )
        self._validate_label_shapes()

        if self.include_raw and not self.stimulus_h5_path.exists():
            raise FileNotFoundError(f"Missing stimulus H5 for raw image output: {self.stimulus_h5_path}")

    def _validate_label_shapes(self) -> None:
        n_labels = max(int(record["label_index"]) for record in self.records) + 1
        if self.caption_text_embeddings.ndim != 3:
            raise ValueError(
                f"Expected caption embeddings shape [labels, captions, dim], got {self.caption_text_embeddings.shape}."
            )
        if self.image_embeddings.ndim != 2:
            raise ValueError(f"Expected image embeddings shape [labels, dim], got {self.image_embeddings.shape}.")
        if self.caption_text_embeddings.shape[0] < n_labels:
            raise ValueError(f"Caption embeddings have {self.caption_text_embeddings.shape[0]} labels, need {n_labels}.")
        if self.image_embeddings.shape[0] < n_labels:
            raise ValueError(f"Image embeddings have {self.image_embeddings.shape[0]} labels, need {n_labels}.")
        if self.include_vae_latents:
            if self.image_vae_latents is None or self.image_vae_latents.ndim != 4:
                shape = None if self.image_vae_latents is None else tuple(self.image_vae_latents.shape)
                raise ValueError(f"Expected VAE latents shape [labels, 4, h, w], got {shape}.")
            if self.image_vae_latents.shape[0] < n_labels:
                raise ValueError(f"VAE latents have {self.image_vae_latents.shape[0]} labels, need {n_labels}.")
        if tuple(self.caption_mask.shape[:2]) != tuple(self.caption_text_embeddings.shape[:2]):
            raise ValueError(
                f"Caption mask shape {tuple(self.caption_mask.shape)} does not match "
                f"caption embeddings {tuple(self.caption_text_embeddings.shape[:2])}."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def _load_beta_volume(self, beta_row: int) -> np.ndarray:
        volume = np.asarray(self.betas[beta_row], dtype=np.float32)
        return normalize_volume(volume, self.normalize)

    def _load_image(self, label_index: int) -> torch.Tensor:
        if self._stimulus_h5 is None:
            self._stimulus_h5 = h5py.File(self.stimulus_h5_path, "r")
        image = np.asarray(self._stimulus_h5["stimuli"][label_index])
        return torch.from_numpy(image.copy()).to(dtype=torch.uint8)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index = self.indices[index]
        record = self.records[record_index]
        label_index = int(record["label_index"])
        fmri = self._load_beta_volume(int(record["beta_row"]))

        item: dict[str, Any] = {
            "fmri": torch.from_numpy(fmri).unsqueeze(0).float(),
            "caption_text_embeddings": self.caption_text_embeddings[label_index].clone(),
            "image_embeddings": self.image_embeddings[label_index].clone(),
            "caption_mask": self.caption_mask[label_index].clone(),
            "metadata": {
                "record_index": record_index,
                "label_index": label_index,
                "global_trial": int(record["global_trial"]),
                "beta_row": int(record["beta_row"]),
                "session": int(record["session"]),
                "trial_in_session": int(record["trial_in_session"]),
                "beta_index": int(record["beta_index"]),
                "rep": int(record["rep"]),
                "nsd_id": int(record["nsd_id"]),
                "coco_id": int(record["coco_id"]),
                "coco_split": str(record["coco_split"]),
            },
        }
        if self.include_vae_latents:
            assert self.image_vae_latents is not None
            item["image_vae_latents"] = self.image_vae_latents[label_index].clone()
        if self.include_raw:
            item["captions"] = list(record["captions"])
            item["image"] = self._load_image(label_index)
        return item


def collate_nsd_concepts(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "fmri": torch.stack([item["fmri"] for item in batch], dim=0),
        "caption_text_embeddings": torch.stack([item["caption_text_embeddings"] for item in batch], dim=0),
        "image_embeddings": torch.stack([item["image_embeddings"] for item in batch], dim=0),
        "caption_mask": torch.stack([item["caption_mask"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
    }
    if "image_vae_latents" in batch[0]:
        out["image_vae_latents"] = torch.stack([item["image_vae_latents"] for item in batch], dim=0)
    if "captions" in batch[0]:
        out["captions"] = [item["captions"] for item in batch]
    if "image" in batch[0]:
        out["image"] = torch.stack([item["image"] for item in batch], dim=0)
    return out


def create_datasets(
    annotation_dir: Path = DEFAULT_ANNOTATION_DIR,
    stim_info_path: Path = DEFAULT_STIM_INFO_PATH,
    betas_path: Path = DEFAULT_BETAS_PATH,
    caption_embeddings_path: Path = DEFAULT_CAPTION_EMBEDDINGS_PATH,
    image_embeddings_path: Path = DEFAULT_IMAGE_EMBEDDINGS_PATH,
    image_vae_latents_path: Path = DEFAULT_IMAGE_VAE_LATENTS_PATH,
    stimulus_h5_path: Path = DEFAULT_STIMULUS_H5_PATH,
    group_key: str = "nsd_id",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
    subject: int = DEFAULT_SUBJECT,
    sessions: int = DEFAULT_SESSIONS,
    trials_per_session: int = DEFAULT_TRIALS_PER_SESSION,
    max_captions: int = DEFAULT_MAX_CAPTIONS,
    normalize: str = "volume",
    include_vae_latents: bool = True,
    include_raw: bool = False,
) -> tuple[NSDConceptDataset, NSDConceptDataset, NSDConceptDataset]:
    records = build_trial_records(
        stim_info_path=stim_info_path,
        annotation_dir=annotation_dir,
        subject=subject,
        sessions=sessions,
        trials_per_session=trials_per_session,
        max_captions=max_captions,
    )
    common = dict(
        annotation_dir=annotation_dir,
        stim_info_path=stim_info_path,
        betas_path=betas_path,
        caption_embeddings_path=caption_embeddings_path,
        image_embeddings_path=image_embeddings_path,
        image_vae_latents_path=image_vae_latents_path,
        stimulus_h5_path=stimulus_h5_path,
        group_key=group_key,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        subject=subject,
        sessions=sessions,
        trials_per_session=trials_per_session,
        max_captions=max_captions,
        normalize=normalize,
        include_vae_latents=include_vae_latents,
        include_raw=include_raw,
        records=records,
    )
    return (
        NSDConceptDataset(split="train", **common),
        NSDConceptDataset(split="val", **common),
        NSDConceptDataset(split="test", **common),
    )


def create_dataloaders(
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle_train: bool = True,
    **dataset_kwargs: Any,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = create_datasets(**dataset_kwargs)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_nsd_concepts,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_nsd_concepts,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_nsd_concepts,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def split_summary(dataset: NSDConceptDataset) -> dict[str, int]:
    records = [dataset.records[idx] for idx in dataset.indices]
    return {
        "trials": len(records),
        "unique_nsd_ids": len({int(record["nsd_id"]) for record in records}),
        "unique_coco_ids": len({int(record["coco_id"]) for record in records}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test NSD label embedding dataloader.")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--stim-info-path", type=Path, default=DEFAULT_STIM_INFO_PATH)
    parser.add_argument("--betas-path", type=Path, default=DEFAULT_BETAS_PATH)
    parser.add_argument("--caption-embeddings-path", type=Path, default=DEFAULT_CAPTION_EMBEDDINGS_PATH)
    parser.add_argument("--image-embeddings-path", type=Path, default=DEFAULT_IMAGE_EMBEDDINGS_PATH)
    parser.add_argument("--image-vae-latents-path", type=Path, default=DEFAULT_IMAGE_VAE_LATENTS_PATH)
    parser.add_argument("--stimulus-h5-path", type=Path, default=DEFAULT_STIMULUS_H5_PATH)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--group-key", default="nsd_id")
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--subject", type=int, default=DEFAULT_SUBJECT)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--trials-per-session", type=int, default=DEFAULT_TRIALS_PER_SESSION)
    parser.add_argument("--max-captions", type=int, default=DEFAULT_MAX_CAPTIONS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normalize", choices=["none", "volume"], default="volume")
    parser.add_argument("--include-vae-latents", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--include-raw", type=str2bool, nargs="?", const=True, default=False)
    args = parser.parse_args()

    train_set, val_set, test_set = create_datasets(
        annotation_dir=args.annotation_dir,
        stim_info_path=args.stim_info_path,
        betas_path=args.betas_path,
        caption_embeddings_path=args.caption_embeddings_path,
        image_embeddings_path=args.image_embeddings_path,
        image_vae_latents_path=args.image_vae_latents_path,
        stimulus_h5_path=args.stimulus_h5_path,
        group_key=args.group_key,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        subject=args.subject,
        sessions=args.sessions,
        trials_per_session=args.trials_per_session,
        max_captions=args.max_captions,
        normalize=args.normalize,
        include_vae_latents=args.include_vae_latents,
        include_raw=args.include_raw,
    )
    datasets = {"train": train_set, "val": val_set, "test": test_set}
    if args.split == "all":
        dataset = NSDConceptDataset(
            annotation_dir=args.annotation_dir,
            stim_info_path=args.stim_info_path,
            betas_path=args.betas_path,
            caption_embeddings_path=args.caption_embeddings_path,
            image_embeddings_path=args.image_embeddings_path,
            image_vae_latents_path=args.image_vae_latents_path,
            stimulus_h5_path=args.stimulus_h5_path,
            split="all",
            group_key=args.group_key,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            subject=args.subject,
            sessions=args.sessions,
            trials_per_session=args.trials_per_session,
            max_captions=args.max_captions,
            normalize=args.normalize,
            include_vae_latents=args.include_vae_latents,
            include_raw=args.include_raw,
            records=train_set.records,
        )
    else:
        dataset = datasets[args.split]

    for name, split_dataset in datasets.items():
        print(name, split_summary(split_dataset))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_nsd_concepts,
    )
    batch = next(iter(loader))
    print(f"selected_split={args.split} samples={len(dataset)}")
    print("fmri", tuple(batch["fmri"].shape), batch["fmri"].dtype)
    print("caption_text_embeddings", tuple(batch["caption_text_embeddings"].shape), batch["caption_text_embeddings"].dtype)
    print("image_embeddings", tuple(batch["image_embeddings"].shape), batch["image_embeddings"].dtype)
    if "image_vae_latents" in batch:
        print("image_vae_latents", tuple(batch["image_vae_latents"].shape), batch["image_vae_latents"].dtype)
    print("caption_mask", tuple(batch["caption_mask"].shape))
    if "image" in batch:
        print("image", tuple(batch["image"].shape), batch["image"].dtype)
    if "captions" in batch:
        print("first captions", batch["captions"][0])
    print("first metadata", batch["metadata"][0])


if __name__ == "__main__":
    main()
