from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoTokenizer, CLIPModel


PROJECT_ROOT = Path(r"D:\PycharmProjects\MindKeyAnimator")
DEFAULT_NSD_ROOT = Path(r"D:\datasets\NSD")
DEFAULT_ANNOTATION_DIR = DEFAULT_NSD_ROOT / "annotations"
DEFAULT_OUTPUT_DIR = DEFAULT_ANNOTATION_DIR / "process"
DEFAULT_PHRASE_CSV = DEFAULT_OUTPUT_DIR / "stop_concepts" / "nsd_subj01_caption_phrases.csv"
DEFAULT_SAVE_PT = PROJECT_ROOT / "save_pt"
DEFAULT_CLIP_DIR = DEFAULT_SAVE_PT / "openai__clip-vit-base-patch32"
DEFAULT_OUTPUT_PREFIX = "nsd_subj01"
SESSION_COUNT = 40
TRIALS_PER_SESSION = 750


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


def split_phrases(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []
    phrases: list[str] = []
    for part in text.split("|||"):
        phrase = normalize_concept(part)
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def split_captions(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []
    captions = [part.strip() for part in text.split("|||")]
    return [caption for caption in captions if caption]


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


def load_phrase_labels(path: Path) -> dict[int, dict[str, Any]]:
    df = pd.read_csv(path)
    required = {"coco_id", "phrases", "captions"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Phrase CSV is missing columns: {sorted(missing)}")

    labels: dict[int, dict[str, Any]] = {}
    for _, row in df.iterrows():
        coco_id = int(row["coco_id"])
        labels[coco_id] = {
            "concepts": split_phrases(row["phrases"]),
            "captions": split_captions(row["captions"]),
            "caption_phrase_count": int(row["phrase_count"]) if "phrase_count" in df.columns else 0,
            "coco_split_from_captions": str(row.get("coco_split_from_captions", "")),
        }
    return labels


def add_concepts(global_concepts: list[str], concepts: list[str]) -> None:
    for concept in concepts:
        if concept not in global_concepts:
            global_concepts.append(concept)


def save_clip_model_locally(model_dir: Path) -> tuple[Any, CLIPModel]:
    model_dir.mkdir(parents=True, exist_ok=True)
    source = str(model_dir) if (model_dir / "config.json").exists() else "openai/clip-vit-base-patch32"
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = CLIPModel.from_pretrained(source)
    if source != str(model_dir):
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    return tokenizer, model


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
    parser = argparse.ArgumentParser(description="Build NSD trial phrase labels and CLIP text features.")
    parser.add_argument("--nsd-root", type=Path, default=DEFAULT_NSD_ROOT)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--phrase-csv", type=Path, default=DEFAULT_PHRASE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--max-sessions", type=int, default=SESSION_COUNT)
    parser.add_argument("--clip-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--limit-trials", type=int, default=0)
    args = parser.parse_args()

    stim_info_path = args.annotation_dir / "nsd_stim_info_merged.csv"
    if not stim_info_path.exists():
        raise FileNotFoundError(f"Missing stimulus index: {stim_info_path}")
    if not args.phrase_csv.exists():
        raise FileNotFoundError(f"Missing phrase CSV: {args.phrase_csv}")
    if not args.clip_dir.exists():
        raise FileNotFoundError(f"Missing local CLIP model directory: {args.clip_dir}")

    output_paths = {
        "labels": args.output_dir / f"{args.output_prefix}_labels.jsonl",
        "features": args.output_dir / f"{args.output_prefix}_features.pt",
        "manifest": args.output_dir / f"{args.output_prefix}_process_manifest.json",
    }
    if not args.overwrite:
        existing = [str(path) for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError("Output files already exist: " + ", ".join(existing))

    print("Loading phrase labels...")
    phrase_labels = load_phrase_labels(args.phrase_csv)

    print("Loading NSD stimulus index...")
    stim_info = pd.read_csv(stim_info_path)
    trial_rows = build_trial_table(stim_info, subject=args.subject, max_sessions=args.max_sessions)
    if args.limit_trials > 0:
        trial_rows = trial_rows[: args.limit_trials]

    image_ids = sorted({int(row["coco_id"]) for row in trial_rows})
    missing_ids = [coco_id for coco_id in image_ids if coco_id not in phrase_labels]
    if missing_ids:
        raise KeyError(f"Phrase CSV is missing {len(missing_ids)} COCO ids, first={missing_ids[:10]}")

    all_concepts: list[str] = []
    for coco_id in image_ids:
        add_concepts(all_concepts, phrase_labels[coco_id]["concepts"])
    concept_to_index = {concept: i for i, concept in enumerate(all_concepts)}

    print("Writing trial label file...")
    output_paths["labels"].parent.mkdir(parents=True, exist_ok=True)
    trial_concept_indices: list[list[int]] = []
    trial_captions: list[list[str]] = []
    global_trials: list[int] = []
    with output_paths["labels"].open("w", encoding="utf-8") as jf:
        for row in trial_rows:
            labels = phrase_labels[int(row["coco_id"])]
            concepts = labels["concepts"]
            concept_indices = [concept_to_index[c] for c in concepts]
            captions = labels["captions"] if labels["captions"] else [" "]
            caption_text = " ".join(captions).strip()
            record = {
                **row,
                "concepts": concepts,
                "caption_phrases": concepts,
                "captions": captions,
                "caption_text": caption_text,
                "caption_phrase_count": labels["caption_phrase_count"],
                "coco_split_from_captions": labels["coco_split_from_captions"],
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            trial_concept_indices.append(concept_indices)
            trial_captions.append(captions)
            global_trials.append(int(row["global_trial"]))

    print("Loading CLIP text model...")
    device = torch.device(args.device)
    clip_tokenizer, clip_model = save_clip_model_locally(args.clip_dir)
    clip_model.to(device)
    clip_model.eval()

    print(f"Encoding {len(all_concepts)} phrase concept CLIP features...")
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
            "concept_source": "caption_phrases",
            "phrase_csv": str(args.phrase_csv),
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
        "concept_source": "caption_phrases",
        "phrase_csv": str(args.phrase_csv),
        "clip_dir": str(args.clip_dir),
        "outputs": {k: str(v) for k, v in output_paths.items()},
    }
    output_paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
