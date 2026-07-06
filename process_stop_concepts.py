from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


DEFAULT_NSD_ROOT = Path(r"D:\datasets\NSD")
DEFAULT_ANNOTATION_DIR = DEFAULT_NSD_ROOT / "annotations"
DEFAULT_OUTPUT_DIR = DEFAULT_ANNOTATION_DIR / "process" / "stop_concepts"
DEFAULT_OUTPUT_PREFIX = "nsd_subj01"
SESSION_COUNT = 40
TRIALS_PER_SESSION = 750


def normalize_token(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9-]", "", text)
    text = text.strip("-")
    return text


def tokenize_caption(text: str, min_token_length: int) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text.lower())
    tokens: list[str] = []
    for token in raw_tokens:
        token = normalize_token(token)
        if len(token) < min_token_length:
            continue
        if token:
            tokens.append(token)
    return tokens


def add_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def load_captions(annotation_dir: Path) -> tuple[dict[int, list[str]], dict[int, str]]:
    captions_by_image: dict[int, list[str]] = defaultdict(list)
    split_by_image: dict[int, str] = {}
    for split in ["train2017", "val2017"]:
        path = annotation_dir / f"captions_{split}.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for image in data["images"]:
            split_by_image[int(image["id"])] = split
        for ann in data["annotations"]:
            caption = str(ann["caption"]).strip()
            if caption:
                captions_by_image[int(ann["image_id"])].append(caption)
    return dict(captions_by_image), split_by_image


def build_unique_subject_images(stim_info: pd.DataFrame, subject: int, max_sessions: int) -> list[dict[str, Any]]:
    rep_columns = [f"subject{subject}_rep{i}" for i in range(3)]
    for col in rep_columns:
        if col not in stim_info.columns:
            raise KeyError(f"Missing required column in nsd_stim_info_merged.csv: {col}")

    max_global_trial = max_sessions * TRIALS_PER_SESSION
    rows: dict[int, dict[str, Any]] = {}
    for _, row in stim_info.iterrows():
        trials = [int(row[col]) for col in rep_columns]
        valid_trials = [trial for trial in trials if 0 < trial <= max_global_trial]
        if not valid_trials:
            continue
        coco_id = int(row["cocoId"])
        rows[coco_id] = {
            "nsd_id": int(row["nsdId"]),
            "coco_id": coco_id,
            "coco_split": str(row["cocoSplit"]),
            "trial_count": len(valid_trials),
            "first_global_trial": min(valid_trials),
        }
    return sorted(rows.values(), key=lambda item: (item["first_global_trial"], item["coco_id"]))


def write_candidate_csvs(
    image_rows: list[dict[str, Any]],
    captions_by_image: dict[int, list[str]],
    split_by_image: dict[int, str],
    output_dir: Path,
    output_prefix: str,
    min_token_length: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stopwords = set(ENGLISH_STOP_WORDS)
    per_image_rows: list[dict[str, Any]] = []
    frequency = Counter()
    image_frequency = Counter()
    examples: dict[str, list[int]] = defaultdict(list)

    for image in image_rows:
        coco_id = int(image["coco_id"])
        captions = captions_by_image.get(coco_id, [])
        raw_words: list[str] = []
        filtered_words: list[str] = []
        for caption in captions:
            for token in tokenize_caption(caption, min_token_length=min_token_length):
                add_unique(raw_words, token)
                if token not in stopwords:
                    add_unique(filtered_words, token)

        frequency.update(filtered_words)
        image_frequency.update(set(filtered_words))
        for word in filtered_words:
            if len(examples[word]) < 5:
                examples[word].append(coco_id)

        per_image_rows.append(
            {
                **image,
                "coco_split_from_captions": split_by_image.get(coco_id, ""),
                "caption_count": len(captions),
                "captions": " ||| ".join(captions),
                "raw_words": " ".join(raw_words),
                "filtered_words": " ".join(filtered_words),
                "filtered_word_count": len(filtered_words),
            }
        )

    frequency_rows = [
        {
            "word": word,
            "total_count": count,
            "image_count": image_frequency[word],
            "example_coco_ids": " ".join(str(x) for x in examples[word]),
        }
        for word, count in frequency.most_common()
    ]

    per_image_path = output_dir / f"{output_prefix}_caption_words.csv"
    frequency_path = output_dir / f"{output_prefix}_caption_word_frequency.csv"
    pd.DataFrame(per_image_rows).to_csv(per_image_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(frequency_rows).to_csv(frequency_path, index=False, encoding="utf-8-sig")
    print(f"saved {per_image_path}")
    print(f"saved {frequency_path}")
    print(f"images={len(per_image_rows)} filtered_vocab={len(frequency_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build caption word CSVs for stop-concept review.")
    parser.add_argument("--nsd-root", type=Path, default=DEFAULT_NSD_ROOT)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--max-sessions", type=int, default=SESSION_COUNT)
    parser.add_argument("--min-token-length", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_dir = args.annotation_dir
    stim_info_path = annotation_dir / "nsd_stim_info_merged.csv"
    if not stim_info_path.exists():
        raise FileNotFoundError(f"Missing stimulus index: {stim_info_path}")

    print("Loading COCO captions...")
    captions_by_image, split_by_image = load_captions(annotation_dir)
    print("Loading NSD stimulus index...")
    stim_info = pd.read_csv(stim_info_path)
    image_rows = build_unique_subject_images(stim_info, subject=args.subject, max_sessions=args.max_sessions)
    write_candidate_csvs(
        image_rows=image_rows,
        captions_by_image=captions_by_image,
        split_by_image=split_by_image,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        min_token_length=args.min_token_length,
    )


if __name__ == "__main__":
    main()
