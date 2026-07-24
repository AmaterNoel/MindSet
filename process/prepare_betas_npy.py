from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.lib.format import open_memmap


DEFAULT_NSD_ROOT = Path(r"D:\datasets\NSD")
DEFAULT_SUBJECT = 1
DEFAULT_SESSIONS = 40
DEFAULT_TRIALS_PER_SESSION = 750
DEFAULT_DTYPE = "float16"
DEFAULT_NSDGENERAL_MASK_NAME = "nsdgeneral.nii.gz"
NSD_S3_BASE_URL = "https://natural-scenes-dataset.s3.amazonaws.com"


def subject_name(subject: int) -> str:
    return f"subj{subject:02d}"


def default_subject_dir(nsd_root: Path, subject: int) -> Path:
    return Path(nsd_root) / subject_name(subject)


def default_source_dir(nsd_root: Path, subject: int) -> Path:
    return default_subject_dir(nsd_root, subject) / "func1pt8mm"


def default_betas_path(nsd_root: Path, subject: int) -> Path:
    return default_subject_dir(nsd_root, subject) / "betas_float16.npy"


def default_nsdgeneral_mask_path(nsd_root: Path, subject: int) -> Path:
    return default_subject_dir(nsd_root, subject) / DEFAULT_NSDGENERAL_MASK_NAME


def default_nsdgeneral_betas_path(nsd_root: Path, subject: int) -> Path:
    return default_subject_dir(nsd_root, subject) / "betas_nsdgeneral_float16.npy"


def nsdgeneral_mask_url(subject: int) -> str:
    name = subject_name(subject)
    return f"{NSD_S3_BASE_URL}/nsddata/ppdata/{name}/func1pt8mm/roi/{DEFAULT_NSDGENERAL_MASK_NAME}"


def parse_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = text.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {text!r}.")


def download_file(url: str, output_path: Path, overwrite: bool) -> None:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"mask exists, skipped download: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(output_path)
    print(f"saved {output_path}")


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
        print(f"betas file exists, skipped NIfTI conversion: {output_path}")
        return

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


def create_nsdgeneral_betas(
    betas_path: Path,
    mask_path: Path,
    output_path: Path,
    chunk_size: int,
    dtype: np.dtype,
    overwrite: bool,
) -> None:
    betas_path = Path(betas_path)
    mask_path = Path(mask_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"nsdgeneral beta file exists, skipped ROI extraction: {output_path}")
        return
    if not betas_path.exists():
        raise FileNotFoundError(f"Missing beta cache: {betas_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing nsdgeneral mask: {mask_path}")

    betas = np.load(betas_path, mmap_mode="r")
    if betas.ndim != 4:
        raise ValueError(f"Expected beta cache shape [trials, x, y, z], got {betas.shape}.")

    mask_img = nib.load(str(mask_path))
    mask = np.asarray(mask_img.dataobj) > 0
    if tuple(mask.shape) != tuple(betas.shape[1:]):
        raise ValueError(f"Mask shape {mask.shape} does not match beta spatial shape {betas.shape[1:]}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = open_memmap(output_path, mode="w+", dtype=dtype, shape=(betas.shape[0], int(mask.sum())))

    for start in range(0, betas.shape[0], chunk_size):
        end = min(start + chunk_size, betas.shape[0])
        out[start:end] = np.asarray(betas[start:end][:, mask], dtype=dtype)
        out.flush()
        print(f"extracted nsdgeneral rows {start:05d}:{end:05d}/{betas.shape[0]}")

    print(f"saved {output_path}")
    print(f"shape={out.shape} dtype={out.dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare NSD beta caches: full 3D betas and nsdgeneral 1D ROI betas."
    )
    parser.add_argument("--nsd-root", type=Path, default=DEFAULT_NSD_ROOT)
    parser.add_argument("--subject", type=int, default=DEFAULT_SUBJECT)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--betas-path", type=Path, default=None)
    parser.add_argument("--nsdgeneral-mask-path", type=Path, default=None)
    parser.add_argument("--nsdgeneral-betas-path", type=Path, default=None)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--trials-per-session", type=int, default=DEFAULT_TRIALS_PER_SESSION)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--dtype", choices=["float16", "float32"], default=DEFAULT_DTYPE)
    parser.add_argument("--overwrite-betas", type=parse_bool, default=False)
    parser.add_argument("--overwrite-mask", type=parse_bool, default=False)
    parser.add_argument("--overwrite-nsdgeneral", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nsd_root = Path(args.nsd_root)
    source_dir = args.source_dir or default_source_dir(nsd_root, args.subject)
    betas_path = args.betas_path or default_betas_path(nsd_root, args.subject)
    mask_path = args.nsdgeneral_mask_path or default_nsdgeneral_mask_path(nsd_root, args.subject)
    nsdgeneral_betas_path = args.nsdgeneral_betas_path or default_nsdgeneral_betas_path(nsd_root, args.subject)
    dtype = np.dtype(args.dtype)

    convert_betas(
        source_dir=source_dir,
        output_path=betas_path,
        sessions=args.sessions,
        trials_per_session=args.trials_per_session,
        chunk_size=args.chunk_size,
        dtype=dtype,
        overwrite=args.overwrite_betas,
    )
    download_file(
        url=nsdgeneral_mask_url(args.subject),
        output_path=mask_path,
        overwrite=args.overwrite_mask,
    )
    create_nsdgeneral_betas(
        betas_path=betas_path,
        mask_path=mask_path,
        output_path=nsdgeneral_betas_path,
        chunk_size=args.chunk_size,
        dtype=dtype,
        overwrite=args.overwrite_nsdgeneral,
    )


if __name__ == "__main__":
    main()
