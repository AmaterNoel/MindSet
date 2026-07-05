from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_NSD_ROOT = Path(r"D:\datasets\NSD")
DEFAULT_BETAS_PATH = DEFAULT_NSD_ROOT / "subj01" / "betas_float16.npy"
DEFAULT_PROCESS_DIR = DEFAULT_NSD_ROOT / "annotations" / "process"
DEFAULT_LABEL_PATH = DEFAULT_PROCESS_DIR / "nsd_subj01_labels.jsonl"
DEFAULT_FEATURE_PATH = DEFAULT_PROCESS_DIR / "nsd_subj01_features.pt"

SplitName = Literal["train", "val", "test", "all"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_feature_file(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


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
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= val_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1), val_ratio must be in [0,1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0 so test split is non-empty.")
    if group_key not in records[0]:
        raise KeyError(f"group_key {group_key!r} is not present in label records.")

    groups = sorted({records[i][group_key] for i in range(len(records))})
    rng = np.random.default_rng(seed)
    groups = [groups[i] for i in rng.permutation(len(groups))]

    n_total = len(groups)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    test_groups = set(groups[n_train + n_val :])

    if split == "train":
        selected = train_groups
    elif split == "val":
        selected = val_groups
    elif split == "test":
        selected = test_groups
    else:
        raise ValueError(f"Unknown split: {split}")

    return [i for i, record in enumerate(records) if record[group_key] in selected]


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
    volume = (volume - mean) / std
    return volume


class NSDConceptDataset(Dataset):
    def __init__(
        self,
        label_path: Path = DEFAULT_LABEL_PATH,
        feature_path: Path = DEFAULT_FEATURE_PATH,
        betas_path: Path = DEFAULT_BETAS_PATH,
        split: SplitName = "train",
        group_key: str = "nsd_id",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        normalize: str = "volume",
        labels: list[dict[str, Any]] | None = None,
        features: dict[str, Any] | None = None,
    ) -> None:
        self.label_path = Path(label_path)
        self.feature_path = Path(feature_path)
        self.betas_path = Path(betas_path)
        self.split = split
        self.group_key = group_key
        self.normalize = normalize

        self.labels = labels if labels is not None else load_jsonl(self.label_path)
        self.features = features if features is not None else load_feature_file(self.feature_path)
        self.indices = deterministic_group_split(
            self.labels,
            split=split,
            group_key=group_key,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        self.concepts: list[str] = list(self.features["concepts"])
        self.concept_clip_features: torch.Tensor = self.features["concept_clip_features"].float()
        self.trial_concept_indices: list[list[int]] = self.features["trial_concept_indices"]
        self.trial_caption_clip_features: torch.Tensor = self.features["trial_caption_clip_features"].float()
        self.trial_caption_mask: torch.Tensor = self.features["trial_caption_mask"].bool()
        self.global_trials: list[int] = [int(x) for x in self.features["global_trials"]]
        self.global_trial_to_feature_row = {trial: i for i, trial in enumerate(self.global_trials)}

        if len(self.labels) != len(self.global_trials):
            raise ValueError(f"Label count {len(self.labels)} != feature trial count {len(self.global_trials)}.")

        if not self.betas_path.exists():
            raise FileNotFoundError(f"Missing beta cache: {self.betas_path}")
        self.betas = np.load(self.betas_path, mmap_mode="r")
        if self.betas.ndim != 4:
            raise ValueError(f"Expected beta cache shape [trials, x, y, z], got {self.betas.shape}.")
        if self.betas.shape[0] < len(self.labels):
            raise ValueError(f"Beta cache has {self.betas.shape[0]} trials but labels contain {len(self.labels)}.")

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _trial_row(session: int, beta_index: int) -> int:
        return (session - 1) * 750 + beta_index

    def _load_beta_volume(self, session: int, beta_index: int) -> np.ndarray:
        row = self._trial_row(session, beta_index)
        volume = np.asarray(self.betas[row], dtype=np.float32)
        volume = normalize_volume(volume, self.normalize)
        return volume

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index = self.indices[index]
        record = self.labels[record_index]
        global_trial = int(record["global_trial"])
        feature_row = self.global_trial_to_feature_row[global_trial]

        concept_indices = self.trial_concept_indices[feature_row]
        concept_features = self.concept_clip_features[concept_indices]
        concept_names = [self.concepts[i] for i in concept_indices]
        fmri = self._load_beta_volume(int(record["session"]), int(record["beta_index"]))

        return {
            "fmri": torch.from_numpy(fmri).unsqueeze(0).float(),
            "caption_clip_features": self.trial_caption_clip_features[feature_row],
            "caption_mask": self.trial_caption_mask[feature_row],
            "concept_clip_features": concept_features,
            "concept_indices": torch.as_tensor(concept_indices, dtype=torch.long),
            "concept_names": concept_names,
            "captions": record.get("captions", []),
            "metadata": {
                "record_index": record_index,
                "global_trial": global_trial,
                "session": int(record["session"]),
                "trial_in_session": int(record["trial_in_session"]),
                "beta_index": int(record["beta_index"]),
                "rep": int(record["rep"]),
                "nsd_id": int(record["nsd_id"]),
                "coco_id": int(record["coco_id"]),
                "coco_split": record["coco_split"],
            },
        }


def collate_nsd_concepts(batch: list[dict[str, Any]]) -> dict[str, Any]:
    fmri = torch.stack([item["fmri"] for item in batch], dim=0)
    caption_clip_features = torch.stack([item["caption_clip_features"] for item in batch], dim=0)
    caption_mask = torch.stack([item["caption_mask"] for item in batch], dim=0)

    max_concepts = max(item["concept_clip_features"].shape[0] for item in batch)
    feature_dim = batch[0]["concept_clip_features"].shape[-1]
    concept_clip_features = torch.zeros(len(batch), max_concepts, feature_dim, dtype=torch.float32)
    concept_mask = torch.zeros(len(batch), max_concepts, dtype=torch.bool)
    concept_indices = torch.full((len(batch), max_concepts), fill_value=-1, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["concept_clip_features"].shape[0]
        concept_clip_features[i, :n] = item["concept_clip_features"]
        concept_mask[i, :n] = True
        concept_indices[i, :n] = item["concept_indices"]

    return {
        "fmri": fmri,
        "caption_clip_features": caption_clip_features,
        "caption_mask": caption_mask,
        "concept_clip_features": concept_clip_features,
        "concept_mask": concept_mask,
        "concept_indices": concept_indices,
        "concept_names": [item["concept_names"] for item in batch],
        "captions": [item["captions"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def create_datasets(
    label_path: Path = DEFAULT_LABEL_PATH,
    feature_path: Path = DEFAULT_FEATURE_PATH,
    betas_path: Path = DEFAULT_BETAS_PATH,
    group_key: str = "nsd_id",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    normalize: str = "volume",
) -> tuple[NSDConceptDataset, NSDConceptDataset, NSDConceptDataset]:
    labels = load_jsonl(label_path)
    features = load_feature_file(feature_path)
    common = dict(
        label_path=label_path,
        feature_path=feature_path,
        betas_path=betas_path,
        group_key=group_key,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        normalize=normalize,
        labels=labels,
        features=features,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test NSD concept dataloader.")
    parser.add_argument("--label-path", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--betas-path", type=Path, default=DEFAULT_BETAS_PATH)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--group-key", default="nsd_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normalize", choices=["none", "volume"], default="volume")
    args = parser.parse_args()

    dataset = NSDConceptDataset(
        label_path=args.label_path,
        feature_path=args.feature_path,
        betas_path=args.betas_path,
        split=args.split,
        group_key=args.group_key,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=args.normalize,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_nsd_concepts,
    )
    batch = next(iter(loader))
    print(f"split={args.split} samples={len(dataset)}")
    print("fmri", tuple(batch["fmri"].shape), batch["fmri"].dtype)
    print("caption_clip_features", tuple(batch["caption_clip_features"].shape))
    print("concept_clip_features", tuple(batch["concept_clip_features"].shape))
    print("concept_mask", tuple(batch["concept_mask"].shape))
    print("first metadata", batch["metadata"][0])
    print("first concepts", batch["concept_names"][0][:20])


if __name__ == "__main__":
    main()
