from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, CLIPModel


PROJECT_ROOT = Path(r"D:\PycharmProjects\MindKeyAnimator")
DEFAULT_NSD_ROOT = Path(r"D:\datasets\NSD")
DEFAULT_ANNOTATION_DIR = DEFAULT_NSD_ROOT / "annotations"
DEFAULT_OUTPUT_DIR = DEFAULT_ANNOTATION_DIR / "process"
DEFAULT_SAVE_PT = PROJECT_ROOT / "save_pt"
DEFAULT_CLIP_DIR = DEFAULT_SAVE_PT / "openai__clip-vit-base-patch32"
DEFAULT_KEYWORD_MODEL_NAME = "vblagoje/bert-english-uncased-finetuned-pos"
DEFAULT_KEYWORD_MODEL_DIR = DEFAULT_SAVE_PT / "vblagoje__bert-english-uncased-finetuned-pos"

DEFAULT_OUTPUT_PREFIX = "nsd_subj01"
SESSION_COUNT = 40
TRIALS_PER_SESSION = 750

CAPTION_POS_LABELS = {"NOUN", "PROPN", "VERB"}
STOP_CONCEPTS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "being",
    "do",
    "does",
    "doing",
    "get",
    "gets",
    "got",
    "have",
    "has",
    "having",
    "image",
    "photo",
    "picture",
    "scene",
    "show",
    "shows",
    "shown",
    "showing",
    "the",
    "there",
    "this",
    "with",
}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def normalize_concept(text: str) -> str:
    text = text.lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(a|an|the)\s+", "", text)
    return text


def add_unique(items: list[str], value: str) -> None:
    value = normalize_concept(value)
    if not value or value in STOP_CONCEPTS:
        return
    if value not in items:
        items.append(value)


def save_hf_model_locally(model_name: str, model_dir: Path, model_type: str) -> tuple[Any, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    has_config = (model_dir / "config.json").exists()
    source = str(model_dir) if has_config else model_name
    tokenizer = AutoTokenizer.from_pretrained(source)
    if model_type == "token-classification":
        model = AutoModelForTokenClassification.from_pretrained(source)
    elif model_type == "clip":
        model = CLIPModel.from_pretrained(source)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    if not has_config:
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    return tokenizer, model


def load_coco_annotations(annotation_dir: Path) -> tuple[dict[int, list[str]], dict[int, list[str]], dict[int, str]]:
    objects_by_image: dict[int, list[str]] = defaultdict(list)
    captions_by_image: dict[int, list[str]] = defaultdict(list)
    split_by_image: dict[int, str] = {}

    for split in ["train2017", "val2017"]:
        instances_path = annotation_dir / f"instances_{split}.json"
        captions_path = annotation_dir / f"captions_{split}.json"
        with instances_path.open("r", encoding="utf-8") as f:
            instances = json.load(f)
        category_name = {int(c["id"]): normalize_concept(c["name"]) for c in instances["categories"]}
        for image in instances["images"]:
            split_by_image[int(image["id"])] = split
        for ann in instances["annotations"]:
            image_id = int(ann["image_id"])
            category = category_name[int(ann["category_id"])]
            add_unique(objects_by_image[image_id], category)

        with captions_path.open("r", encoding="utf-8") as f:
            captions = json.load(f)
        for ann in captions["annotations"]:
            caption = str(ann["caption"]).strip()
            if caption:
                captions_by_image[int(ann["image_id"])].append(caption)

    return dict(objects_by_image), dict(captions_by_image), split_by_image


def merge_wordpiece_tokens(tokens: list[str], labels: list[str]) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    for token, label in zip(tokens, labels):
        if token.startswith("##") and merged:
            prev_token, prev_label = merged[-1]
            merged[-1] = (prev_token + token[2:], prev_label)
        else:
            merged.append((token, label))
    return merged


def extract_keywords_from_caption(
    caption: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification,
    device: torch.device,
) -> list[str]:
    encoded = tokenizer(caption, return_tensors="pt", truncation=True, max_length=96)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits[0]
    pred = logits.argmax(dim=-1).detach().cpu().tolist()
    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    words: list[str] = []
    labels: list[str] = []
    for token, label_id in zip(tokens, pred):
        if token in {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}:
            continue
        token = token.strip()
        if not token:
            continue
        label = model.config.id2label[int(label_id)]
        words.append(token)
        labels.append(label)

    merged = merge_wordpiece_tokens(words, labels)
    concepts: list[str] = []
    noun_phrase: list[str] = []

    def flush_noun_phrase() -> None:
        if noun_phrase:
            add_unique(concepts, " ".join(noun_phrase))
            noun_phrase.clear()

    for word, label in merged:
        word = normalize_concept(word)
        if not word or word in STOP_CONCEPTS:
            flush_noun_phrase()
            continue
        if label in {"NOUN", "PROPN"}:
            noun_phrase.append(word)
            continue
        flush_noun_phrase()
        if label == "VERB":
            add_unique(concepts, word)

    flush_noun_phrase()
    return concepts


def extract_keywords_for_captions(
    captions: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification,
    device: torch.device,
) -> list[str]:
    keywords: list[str] = []
    for caption in captions:
        for keyword in extract_keywords_from_caption(caption, tokenizer, model, device):
            add_unique(keywords, keyword)
    return keywords


def build_trial_table(stim_info: pd.DataFrame, subject: int, max_sessions: int) -> list[dict[str, Any]]:
    rep_columns = [f"subject{subject}_rep{i}" for i in range(3)]
    for col in rep_columns:
        if col not in stim_info.columns:
            raise KeyError(f"Missing required column in nsd_stim_info_merged.csv: {col}")

    max_global_trial = max_sessions * TRIALS_PER_SESSION
    rows: list[dict[str, Any]] = []
    for _, row in stim_info.iterrows():
        for rep_idx, col in enumerate(rep_columns):
            global_trial = int(row[col])
            if global_trial <= 0 or global_trial > max_global_trial:
                continue
            rows.append(
                {
                    "global_trial": global_trial,
                    "session": (global_trial - 1) // TRIALS_PER_SESSION + 1,
                    "trial_in_session": (global_trial - 1) % TRIALS_PER_SESSION + 1,
                    "beta_index": (global_trial - 1) % TRIALS_PER_SESSION,
                    "rep": rep_idx,
                    "nsd_id": int(row["nsdId"]),
                    "coco_id": int(row["cocoId"]),
                    "coco_split": str(row["cocoSplit"]),
                }
            )
    rows.sort(key=lambda x: x["global_trial"])
    expected = list(range(1, max_global_trial + 1))
    got = [int(row["global_trial"]) for row in rows]
    if got != expected:
        missing = sorted(set(expected) - set(got))[:20]
        duplicated = sorted(k for k, v in pd.Series(got).value_counts().items() if v > 1)[:20]
        raise RuntimeError(
            "Subject trial table is not a complete contiguous sequence. "
            f"rows={len(rows)} expected={max_global_trial} missing_head={missing} duplicated_head={duplicated}"
        )
    return rows


def encode_clip_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: CLIPModel,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            text_outputs = model.text_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
            )
            feat = model.text_projection(text_outputs.pooler_output)
            feat = torch.nn.functional.normalize(feat, dim=-1)
        features.append(feat.detach().cpu().to(torch.float32))
    return torch.cat(features, dim=0) if features else torch.empty(0, 512, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NSD trial concept labels and CLIP text features.")
    parser.add_argument("--nsd-root", type=Path, default=DEFAULT_NSD_ROOT)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-pt-dir", type=Path, default=DEFAULT_SAVE_PT)
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP_DIR)
    parser.add_argument("--keyword-model-name", default=DEFAULT_KEYWORD_MODEL_NAME)
    parser.add_argument("--keyword-model-dir", type=Path, default=DEFAULT_KEYWORD_MODEL_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--max-sessions", type=int, default=SESSION_COUNT)
    parser.add_argument("--clip-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--limit-trials", type=int, default=0)
    args = parser.parse_args()

    annotation_dir = args.annotation_dir
    output_dir = args.output_dir
    stim_info_path = annotation_dir / "nsd_stim_info_merged.csv"
    if not stim_info_path.exists():
        raise FileNotFoundError(f"Missing stimulus index: {stim_info_path}")
    if not args.clip_dir.exists():
        raise FileNotFoundError(f"Missing local CLIP model directory: {args.clip_dir}")

    output_paths = {
        "labels": output_dir / f"{args.output_prefix}_labels.jsonl",
        "features": output_dir / f"{args.output_prefix}_features.pt",
        "manifest": output_dir / f"{args.output_prefix}_process_manifest.json",
    }
    if not args.overwrite:
        existing = [str(path) for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError("Output files already exist: " + ", ".join(existing))

    print("Loading COCO annotations...")
    objects_by_image, captions_by_image, split_by_image = load_coco_annotations(annotation_dir)

    print("Loading NSD stimulus index...")
    stim_info = pd.read_csv(stim_info_path)
    trial_rows = build_trial_table(stim_info, subject=args.subject, max_sessions=args.max_sessions)
    if args.limit_trials > 0:
        trial_rows = trial_rows[: args.limit_trials]

    print("Loading keyword POS model...")
    device = torch.device(args.device)
    keyword_tokenizer, keyword_model = save_hf_model_locally(
        args.keyword_model_name, args.keyword_model_dir, "token-classification"
    )
    keyword_model.to(device)
    keyword_model.eval()

    image_ids = sorted({int(row["coco_id"]) for row in trial_rows})
    image_label_cache: dict[int, dict[str, Any]] = {}
    all_concepts: list[str] = []

    print(f"Extracting caption keywords for {len(image_ids)} unique images...")
    for idx, coco_id in enumerate(image_ids, start=1):
        coco_objects = objects_by_image.get(coco_id, [])
        captions = captions_by_image.get(coco_id, [])
        caption_keywords = extract_keywords_for_captions(captions, keyword_tokenizer, keyword_model, device)
        concepts: list[str] = []
        for concept in coco_objects:
            add_unique(concepts, concept)
        for concept in caption_keywords:
            add_unique(concepts, concept)
        for concept in concepts:
            add_unique(all_concepts, concept)
        image_label_cache[coco_id] = {
            "coco_objects": coco_objects,
            "caption_keywords": caption_keywords,
            "concepts": concepts,
            "captions": captions,
            "coco_split_from_annotations": split_by_image.get(coco_id, ""),
        }
        if idx % 500 == 0 or idx == len(image_ids):
            print(f"  processed {idx}/{len(image_ids)} images")

    concept_to_index = {concept: i for i, concept in enumerate(all_concepts)}

    print("Writing trial label file...")
    output_paths["labels"].parent.mkdir(parents=True, exist_ok=True)
    trial_concept_indices: list[list[int]] = []
    trial_captions: list[list[str]] = []
    global_trials: list[int] = []
    with output_paths["labels"].open("w", encoding="utf-8") as jf:
        for row in trial_rows:
            labels = image_label_cache[int(row["coco_id"])]
            concepts = labels["concepts"]
            concept_indices = [concept_to_index[c] for c in concepts]
            caption_text = " ".join(labels["captions"]).strip()
            record = {
                **row,
                "concepts": concepts,
                "coco_objects": labels["coco_objects"],
                "caption_keywords": labels["caption_keywords"],
                "captions": labels["captions"],
                "caption_text": caption_text,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            trial_concept_indices.append(concept_indices)
            captions = labels["captions"] if labels["captions"] else [" "]
            trial_captions.append(captions)
            global_trials.append(int(row["global_trial"]))

    print("Loading CLIP text model...")
    clip_tokenizer, clip_model = save_hf_model_locally("openai/clip-vit-base-patch32", args.clip_dir, "clip")
    clip_model.to(device)
    clip_model.eval()

    print(f"Encoding {len(all_concepts)} concept CLIP features...")
    concept_features = encode_clip_texts(all_concepts, clip_tokenizer, clip_model, device, args.clip_batch_size)

    max_captions = max((len(captions) for captions in trial_captions), default=0)
    flat_caption_texts: list[str] = []
    caption_mask = torch.zeros(len(trial_captions), max_captions, dtype=torch.bool)
    for trial_idx, captions in enumerate(trial_captions):
        for caption_idx in range(max_captions):
            if caption_idx < len(captions):
                flat_caption_texts.append(captions[caption_idx])
                caption_mask[trial_idx, caption_idx] = True
            else:
                flat_caption_texts.append(" ")

    print(f"Encoding {len(flat_caption_texts)} individual caption CLIP features...")
    flat_caption_features = encode_clip_texts(
        flat_caption_texts, clip_tokenizer, clip_model, device, args.clip_batch_size
    )
    caption_features = flat_caption_features.view(len(trial_captions), max_captions, -1)
    torch.save(
        {
            "concepts": all_concepts,
            "concept_to_index": concept_to_index,
            "concept_clip_features": concept_features,
            "trial_concept_indices": trial_concept_indices,
            "trial_caption_clip_features": caption_features,
            "trial_caption_mask": caption_mask,
            "global_trials": global_trials,
            "feature_type": "normalized_clip_text_features",
            "clip_model": str(args.clip_dir),
            "caption_source": "individual_coco_captions_per_trial",
        },
        output_paths["features"],
    )

    manifest = {
        "subject": args.subject,
        "max_sessions": args.max_sessions,
        "trials_per_session": TRIALS_PER_SESSION,
        "trial_count": len(trial_rows),
        "unique_image_count": len(image_ids),
        "concept_count": len(all_concepts),
        "caption_pos_labels": sorted(CAPTION_POS_LABELS),
        "keyword_model": args.keyword_model_name,
        "keyword_model_dir": str(args.keyword_model_dir),
        "clip_dir": str(args.clip_dir),
        "outputs": {k: str(v) for k, v in output_paths.items()},
    }
    output_paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
