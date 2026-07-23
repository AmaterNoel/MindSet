from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_NSD_ROOT = Path(r"D:\datasets\NSD")
DEFAULT_IMAGE_SIZE = 256
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
        description="Download COCO images and build an NSD subject stimulus H5 in label order."
    )
    parser.add_argument("--subject", default="S1", help="Subject id, e.g. S1, S2, 1, or subj01.")
    parser.add_argument("--nsd-root", type=Path, default=DEFAULT_NSD_ROOT)
    parser.add_argument("--annotation-dir", type=Path, default=None)
    parser.add_argument("--stim-info", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to wait between failed download retries.")
    parser.add_argument("--limit-records", type=int, default=0, help="Only process the first N records for smoke tests.")
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="none")
    parser.add_argument("--overwrite", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--resume", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument(
        "--on-error",
        choices=("raise", "blank"),
        default="raise",
        help="Use blank images for failed downloads only when explicitly requested.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    subject_name, subject_number, subject_dir_name = parse_subject(args.subject)
    args.subject_name = subject_name
    args.subject_number = subject_number
    args.subject_dir_name = subject_dir_name
    args.annotation_dir = args.annotation_dir or args.nsd_root / "annotations"
    args.stim_info = args.stim_info or args.annotation_dir / "nsd_stim_info_merged.csv"
    args.output = args.output or args.nsd_root / "stimulus" / f"{subject_name}_stimuli_label_order_256.h5py"
    return args


def load_coco_images(annotation_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    images: dict[tuple[str, int], dict[str, Any]] = {}
    for split in COCO_SPLITS:
        path = annotation_dir / f"captions_{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing COCO caption file: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for image in data.get("images", []):
            image_id = int(image["id"])
            images[(split, image_id)] = {
                "id": image_id,
                "file_name": str(image.get("file_name", "")),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "coco_url": str(image.get("coco_url", "")),
                "flickr_url": str(image.get("flickr_url", "")),
            }
    return images


def parse_crop_box(value: str) -> tuple[float, float, float, float]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, tuple) or len(parsed) != 4:
        raise ValueError(f"Invalid cropBox value: {value!r}")
    top, bottom, left, right = (float(x) for x in parsed)
    return top, bottom, left, right


def load_subject_records(
    stim_info: Path,
    subject_number: int,
    coco_images: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    if not stim_info.exists():
        raise FileNotFoundError(f"Missing NSD stimulus info CSV: {stim_info}")

    subject_col = f"subject{subject_number}"
    with stim_info.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required = {"nsdId", "cocoId", "cocoSplit", "cropBox", subject_col}
    missing = required - fieldnames
    if missing:
        raise KeyError(f"Stimulus CSV is missing columns: {sorted(missing)}")

    records: list[dict[str, Any]] = []
    for split in COCO_SPLITS:
        for row in rows:
            if row["cocoSplit"] != split or int(row[subject_col]) != 1:
                continue
            coco_id = int(row["cocoId"])
            image_meta = coco_images.get((split, coco_id))
            if image_meta is None:
                raise KeyError(f"Missing COCO image metadata for {split}/{coco_id}")
            records.append(
                {
                    "h5_index": len(records),
                    "nsd_id": int(row["nsdId"]),
                    "coco_id": coco_id,
                    "coco_split": split,
                    "crop_box": parse_crop_box(row["cropBox"]),
                    "file_name": image_meta["file_name"],
                    "width": image_meta["width"],
                    "height": image_meta["height"],
                    "coco_url": image_meta["coco_url"],
                    "flickr_url": image_meta["flickr_url"],
                }
            )
    return records


def normalize_url(url: str) -> str:
    return url.strip()


def download_bytes(urls: list[str], timeout: float, retries: int, sleep: float) -> bytes:
    last_error: Exception | None = None
    headers = {"User-Agent": "MindKeyAnimator/prepare_stimulus.py"}
    cleaned_urls = [normalize_url(url) for url in urls if url]
    for attempt in range(1, retries + 1):
        for url in cleaned_urls:
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return response.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
        if attempt < retries:
            time.sleep(sleep)
    raise RuntimeError(f"Failed to download image from {cleaned_urls}") from last_error


def crop_and_resize(image: Image.Image, crop_box: tuple[float, float, float, float], size: int) -> np.ndarray:
    image = image.convert("RGB")
    width, height = image.size
    top_frac, bottom_frac, left_frac, right_frac = crop_box
    left = int(round(left_frac * width))
    upper = int(round(top_frac * height))
    right = int(round(width - right_frac * width))
    lower = int(round(height - bottom_frac * height))
    if right <= left or lower <= upper:
        raise ValueError(f"Invalid pixel crop box {(left, upper, right, lower)} for image size {(width, height)}")
    cropped = image.crop((left, upper, right, lower))
    resized = cropped.resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.uint8)
    return np.transpose(array, (2, 0, 1))


def make_string_dataset(h5: h5py.File, name: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)


def progress_iter(records: list[dict[str, Any]], total: int | None = None, initial: int = 0, desc: str = "downloading"):
    if tqdm is not None:
        yield from tqdm(records, desc=desc, unit="image", total=total, initial=initial)
        return
    total = total or len(records)
    for index, record in enumerate(records, start=1):
        done = initial + index
        if index == 1 or done % 100 == 0 or done == total:
            print(f"{desc} {done}/{total}")
        yield record


def create_output_datasets(
    h5: h5py.File,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    compression: str | None,
) -> None:
    h5.attrs["subject"] = args.subject_name
    h5.attrs["order"] = "train2017 CSV order followed by val2017 CSV order"
    h5.attrs["image_size"] = args.image_size
    h5.attrs["stim_info"] = str(args.stim_info)
    h5.attrs["annotation_dir"] = str(args.annotation_dir)

    h5.create_dataset(
        "stimuli",
        shape=(len(records), 3, args.image_size, args.image_size),
        dtype=np.uint8,
        chunks=(1, 3, args.image_size, args.image_size),
        compression=compression,
    )
    h5.create_dataset("nsd_id", data=np.asarray([r["nsd_id"] for r in records], dtype=np.int64))
    h5.create_dataset("coco_id", data=np.asarray([r["coco_id"] for r in records], dtype=np.int64))
    make_string_dataset(h5, "coco_split", [r["coco_split"] for r in records])
    make_string_dataset(h5, "file_name", [r["file_name"] for r in records])
    make_string_dataset(h5, "coco_url", [r["coco_url"] for r in records])
    h5.create_dataset("crop_box", data=np.asarray([r["crop_box"] for r in records], dtype=np.float32))
    h5.create_dataset("written", data=np.zeros(len(records), dtype=np.bool_))


def ensure_written_dataset(h5: h5py.File) -> h5py.Dataset:
    if "written" in h5:
        return h5["written"]

    stimuli = h5["stimuli"]
    written = h5.create_dataset("written", shape=(stimuli.shape[0],), dtype=np.bool_)
    for index in progress_iter(
        [{"h5_index": i} for i in range(stimuli.shape[0])],
        total=stimuli.shape[0],
        desc="scanning existing",
    ):
        h5_index = int(index["h5_index"])
        written[h5_index] = int(stimuli[h5_index].sum()) != 0
    h5.flush()
    return written


def validate_resume_file(h5: h5py.File, args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    if "stimuli" not in h5:
        raise KeyError("Resume file is missing dataset 'stimuli'.")
    expected_shape = (len(records), 3, args.image_size, args.image_size)
    if tuple(h5["stimuli"].shape) != expected_shape:
        raise RuntimeError(f"Resume file shape {tuple(h5['stimuli'].shape)} does not match {expected_shape}.")
    if "nsd_id" in h5:
        for index in (0, len(records) - 1):
            if int(h5["nsd_id"][index]) != int(records[index]["nsd_id"]):
                raise RuntimeError("Resume file order does not match current records.")


def write_stimulus_h5(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    compression = None if args.compression == "none" else args.compression
    temp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists. Pass --overwrite to replace it.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    should_resume = temp_path.exists() and args.resume
    if temp_path.exists() and not should_resume:
        temp_path.unlink()

    with h5py.File(temp_path, "r+" if should_resume else "w") as h5:
        if should_resume:
            validate_resume_file(h5, args, records)
            print(f"resuming from: {temp_path}")
        else:
            create_output_datasets(h5, args, records, compression)

        stimuli = h5["stimuli"]
        written = ensure_written_dataset(h5)
        completed = int(np.asarray(written[:], dtype=np.bool_).sum())
        pending_records = [record for record in records if not bool(written[int(record["h5_index"])])]
        if completed:
            print(f"already written: {completed}/{len(records)}")
        failed: list[str] = []
        for record in progress_iter(pending_records, total=len(records), initial=completed):
            try:
                content = download_bytes(
                    [record["coco_url"], record["flickr_url"]],
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                )
                image = Image.open(io.BytesIO(content))
                stimuli[record["h5_index"]] = crop_and_resize(image, record["crop_box"], args.image_size)
                written[record["h5_index"]] = True
                if (int(record["h5_index"]) + 1) % 50 == 0:
                    h5.flush()
            except Exception as exc:
                if args.on_error == "raise":
                    h5.flush()
                    raise RuntimeError(
                        f"Failed at h5_index={record['h5_index']} nsd_id={record['nsd_id']} "
                        f"coco_id={record['coco_id']} split={record['coco_split']}"
                    ) from exc
                failed.append(f"{record['h5_index']},{record['nsd_id']},{record['coco_id']},{record['coco_split']}")
                stimuli[record["h5_index"]] = np.zeros((3, args.image_size, args.image_size), dtype=np.uint8)
                written[record["h5_index"]] = True

        if "failed_downloads" in h5:
            del h5["failed_downloads"]
        make_string_dataset(h5, "failed_downloads", failed)
        h5.flush()

    if args.output.exists():
        args.output.unlink()
    temp_path.replace(args.output)


def main() -> None:
    args = resolve_paths(parse_args())
    coco_images = load_coco_images(args.annotation_dir)
    records = load_subject_records(args.stim_info, args.subject_number, coco_images)
    if args.limit_records > 0:
        records = records[: args.limit_records]
    if not records:
        raise RuntimeError(f"No stimulus records found for {args.subject_name}")

    print(f"subject: {args.subject_name}")
    print(f"records: {len(records)}")
    print(f"output: {args.output}")
    print("first records:")
    for record in records[:5]:
        print(
            f"  index={record['h5_index']} nsd_id={record['nsd_id']} "
            f"coco_id={record['coco_id']} split={record['coco_split']} file={record['file_name']}"
        )

    write_stimulus_h5(args, records)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
