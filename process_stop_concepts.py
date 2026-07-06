from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


DEFAULT_NSD_ROOT = Path(r"D:\datasets\NSD")
DEFAULT_ANNOTATION_DIR = DEFAULT_NSD_ROOT / "annotations"
DEFAULT_OUTPUT_DIR = DEFAULT_ANNOTATION_DIR / "process" / "stop_concepts"
DEFAULT_OUTPUT_PREFIX = "nsd_subj01"
DEFAULT_SPACY_MODEL = "en_core_web_sm"
SESSION_COUNT = 40
TRIALS_PER_SESSION = 750
MAX_VERB_PHRASE_WORDS = 6

PHRASE_STOP_HEADS = {
    "image",
    "photo",
    "picture",
    "scene",
    "view",
}
KEEP_SINGLE_POS = {"NOUN", "PROPN"}
VERB_POS = {"VERB"}
OBJECT_DEPS = {"dobj", "obj", "attr", "oprd", "pobj"}
PREP_DEPS = {"prep"}


def ensure_spacy_model(model_name: str):
    try:
        import spacy
    except ImportError:
        print("Installing spaCy...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
        import spacy

    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model {model_name}...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        return spacy.load(model_name)


def normalize_phrase(text: str) -> str:
    text = text.lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(a|an|the)\s+", "", text)
    return text


def is_stop_phrase(text: str, stopwords: set[str], min_phrase_chars: int) -> bool:
    text = normalize_phrase(text)
    if len(text) < min_phrase_chars:
        return True
    words = text.split()
    if not words:
        return True
    if all(word in stopwords for word in words):
        return True
    if words[-1] in PHRASE_STOP_HEADS:
        return True
    if words[-1] in {"and", "or", "with", "of", "in", "on", "under", "over", "near", "beside"}:
        return True
    return False


def add_unique(items: list[str], value: str, stopwords: set[str], min_phrase_chars: int) -> None:
    value = normalize_phrase(value)
    if is_stop_phrase(value, stopwords, min_phrase_chars):
        return
    if value not in items:
        items.append(value)


def token_subtree_text(token: Any) -> str:
    subtree = sorted(token.subtree, key=lambda x: x.i)
    return " ".join(t.text for t in subtree)


def add_verb_phrase(items: list[str], value: str, stopwords: set[str], min_phrase_chars: int) -> None:
    value = normalize_phrase(value)
    if len(value.split()) > MAX_VERB_PHRASE_WORDS:
        return
    add_unique(items, value, stopwords, min_phrase_chars)


def noun_chunk_core(chunk: Any) -> str:
    tokens = [t for t in chunk if not (t.dep_ == "det" or t.text.lower() in {"a", "an", "the"})]
    return " ".join(t.text for t in tokens)


def noun_chunk_with_of_phrases(chunk: Any) -> list[str]:
    phrases = [noun_chunk_core(chunk)]
    root = chunk.root
    for prep in root.children:
        if prep.dep_ != "prep" or prep.text.lower() != "of":
            continue
        for pobj in prep.children:
            if pobj.dep_ not in OBJECT_DEPS:
                continue
            phrases.append(f"{noun_chunk_core(chunk)} of {token_subtree_text(pobj)}")
    return phrases


def extract_caption_phrases(doc: Any, stopwords: set[str], min_phrase_chars: int) -> list[str]:
    phrases: list[str] = []

    for chunk in doc.noun_chunks:
        for phrase in noun_chunk_with_of_phrases(chunk):
            add_unique(phrases, phrase, stopwords, min_phrase_chars)

    for token in doc:
        if token.pos_ not in KEEP_SINGLE_POS:
            continue
        text = normalize_phrase(token.lemma_ if token.lemma_ != "-PRON-" else token.text)
        if text in stopwords or text in PHRASE_STOP_HEADS:
            continue
        add_unique(phrases, text, stopwords, min_phrase_chars)

    for verb in doc:
        if verb.pos_ not in VERB_POS:
            continue
        verb_text = normalize_phrase(verb.text)
        if not verb_text or verb_text in stopwords:
            continue

        subjects = [
            child
            for child in verb.children
            if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {"NOUN", "PROPN", "PRON"}
        ]
        objects = [child for child in verb.children if child.dep_ in OBJECT_DEPS]
        preps = [child for child in verb.children if child.dep_ in PREP_DEPS]

        for obj in objects:
            obj_text = token_subtree_text(obj)
            add_verb_phrase(phrases, f"{verb_text} {obj_text}", stopwords, min_phrase_chars)
            for subj in subjects:
                add_verb_phrase(phrases, f"{token_subtree_text(subj)} {verb_text} {obj_text}", stopwords, min_phrase_chars)

        for prep in preps:
            for pobj in prep.children:
                if pobj.dep_ not in OBJECT_DEPS:
                    continue
                pobj_text = token_subtree_text(pobj)
                add_verb_phrase(phrases, f"{verb_text} {prep.text} {pobj_text}", stopwords, min_phrase_chars)
                for subj in subjects:
                    add_verb_phrase(
                        phrases,
                        f"{token_subtree_text(subj)} {verb_text} {prep.text} {pobj_text}",
                        stopwords,
                        min_phrase_chars,
                    )

    return phrases


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


def write_phrase_csvs(
    image_rows: list[dict[str, Any]],
    captions_by_image: dict[int, list[str]],
    split_by_image: dict[int, str],
    output_dir: Path,
    output_prefix: str,
    spacy_model: str,
    min_phrase_chars: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.glob(f"{output_prefix}_caption_word*.csv"):
        old_path.unlink()

    nlp = ensure_spacy_model(spacy_model)
    stopwords = set(ENGLISH_STOP_WORDS)
    per_image_rows: list[dict[str, Any]] = []
    frequency = Counter()
    image_frequency = Counter()
    examples: dict[str, list[int]] = defaultdict(list)

    for idx, image in enumerate(image_rows, start=1):
        coco_id = int(image["coco_id"])
        captions = captions_by_image.get(coco_id, [])
        phrases: list[str] = []
        for doc in nlp.pipe(captions, batch_size=64):
            for phrase in extract_caption_phrases(doc, stopwords, min_phrase_chars):
                add_unique(phrases, phrase, stopwords, min_phrase_chars)

        frequency.update(phrases)
        image_frequency.update(set(phrases))
        for phrase in phrases:
            if len(examples[phrase]) < 5:
                examples[phrase].append(coco_id)

        per_image_rows.append(
            {
                **image,
                "coco_split_from_captions": split_by_image.get(coco_id, ""),
                "caption_count": len(captions),
                "captions": " ||| ".join(captions),
                "phrases": " ||| ".join(phrases),
                "phrase_count": len(phrases),
            }
        )
        if idx % 500 == 0 or idx == len(image_rows):
            print(f"  processed {idx}/{len(image_rows)} images")

    frequency_rows = [
        {
            "phrase": phrase,
            "total_count": count,
            "image_count": image_frequency[phrase],
            "word_count": len(phrase.split()),
            "example_coco_ids": " ".join(str(x) for x in examples[phrase]),
        }
        for phrase, count in frequency.most_common()
    ]

    per_image_path = output_dir / f"{output_prefix}_caption_phrases.csv"
    frequency_path = output_dir / f"{output_prefix}_caption_phrase_frequency.csv"
    pd.DataFrame(per_image_rows).to_csv(per_image_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(frequency_rows).to_csv(frequency_path, index=False, encoding="utf-8-sig")
    print(f"saved {per_image_path}")
    print(f"saved {frequency_path}")
    print(f"images={len(per_image_rows)} phrase_vocab={len(frequency_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build caption phrase CSVs for concept review.")
    parser.add_argument("--nsd-root", type=Path, default=DEFAULT_NSD_ROOT)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--max-sessions", type=int, default=SESSION_COUNT)
    parser.add_argument("--min-phrase-chars", type=int, default=2)
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
    print("Extracting concept phrases...")
    write_phrase_csvs(
        image_rows=image_rows,
        captions_by_image=captions_by_image,
        split_by_image=split_by_image,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        spacy_model=args.spacy_model,
        min_phrase_chars=args.min_phrase_chars,
    )


if __name__ == "__main__":
    main()
