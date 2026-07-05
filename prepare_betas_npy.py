from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.lib.format import open_memmap


DEFAULT_NSD_ROOT = Path(r"D:\datasets\NSD")
DEFAULT_SOURCE_DIR = DEFAULT_NSD_ROOT / "subj01" / "func1pt8mm"
DEFAULT_OUTPUT_PATH = DEFAULT_NSD_ROOT / "subj01" / "betas_float16.npy"


def parse_bool(text: str) -> bool:
    value = text.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {text!r}.")


def convert_betas(
    source_dir: Path,
    output_path: Path,
    sessions: int,
    trials_per_session: int,
    chunk_size: int,
    dtype: np.dtype,
    overwrite: bool,
) -> None:
    source_dir = Path(source_dir)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Set --overwrite true to rebuild it.")

    first_path = source_dir / "betas_session01.nii.gz"
    if not first_path.exists():
        raise FileNotFoundError(f"Missing source beta file: {first_path}")

    first_img = nib.load(str(first_path))
    spatial_shape = first_img.shape[:3]
    if first_img.shape[3] != trials_per_session:
        raise ValueError(f"{first_path} has {first_img.shape[3]} trials, expected {trials_per_session}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_trials = sessions * trials_per_session
    betas = open_memmap(output_path, mode="w+", dtype=dtype, shape=(total_trials, *spatial_shape))

    for session in range(1, sessions + 1):
        beta_path = source_dir / f"betas_session{session:02d}.nii.gz"
        if not beta_path.exists():
            raise FileNotFoundError(f"Missing source beta file: {beta_path}")
        img = first_img if session == 1 else nib.load(str(beta_path))
        if img.shape[:3] != spatial_shape or img.shape[3] != trials_per_session:
            raise ValueError(f"{beta_path} shape {img.shape} does not match expected {(*spatial_shape, trials_per_session)}.")

        base_row = (session - 1) * trials_per_session
        for start in range(0, trials_per_session, chunk_size):
            end = min(start + chunk_size, trials_per_session)
            block = np.asarray(img.dataobj[..., start:end])
            block = np.moveaxis(block, -1, 0).astype(dtype, copy=False)
            betas[base_row + start : base_row + end] = block
        betas.flush()
        print(f"converted session {session:02d}/{sessions}")

    print(f"saved {output_path}")
    print(f"shape={betas.shape} dtype={betas.dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NSD beta NIfTI sessions into one mmap-friendly .npy file.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--trials-per-session", type=int, default=750)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_betas(
        source_dir=args.source_dir,
        output_path=args.output_path,
        sessions=args.sessions,
        trials_per_session=args.trials_per_session,
        chunk_size=args.chunk_size,
        dtype=np.dtype(args.dtype),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
