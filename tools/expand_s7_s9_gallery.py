from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data0/home/longnuoer/miniconda3/envs/lne3.12/bin/python"
OUTPUT = ROOT / "output" / "semantic_model"
TEMP = ROOT / "output_smoke" / "semantic_50"
STATUS = OUTPUT / "_expand_s7_s9_50_status.json"


def write_status(stage: str, **extra: object) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds"), **extra}
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def run(command: list[str], log_path: Path, gpu: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def validate_gallery(run_dir: Path, expected: int = 50) -> None:
    counts = {
        split: len(list((run_dir / "gallery" / split).glob("*_comparison.png")))
        for split in ("train", "val", "test")
    }
    if any(count < expected for count in counts.values()):
        raise RuntimeError(f"Incomplete gallery in {run_dir}: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot S7/S9 gallery expansion to 50 samples per split.")
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=5)
    args = parser.parse_args()

    s7 = OUTPUT / "expS7_caption_minloss_only_30ep"
    s9 = OUTPUT / "expS9_image_weak_multipos_30ep"
    s9_temp = TEMP / "retrainS9_for_50samples"
    TEMP.mkdir(parents=True, exist_ok=True)

    try:
        write_status("waiting_for_s7_generation", pid=args.wait_pid)
        while process_exists(args.wait_pid):
            time.sleep(60)
        validate_gallery(s7)

        write_status("evaluating_s7")
        run(
            [PYTHON, "models/evaluate_semantic_gallery.py", "--run-dir", str(s7), "--device", "cuda"],
            s7 / "evaluation50.log",
            args.gpu,
        )

        write_status("training_s9")
        run(
            [
                PYTHON,
                "models/train_base_model_1D.py",
                "--epochs", "30",
                "--eval-every", "1",
                "--batch-size", "256",
                "--run-name", s9_temp.name,
                "--output-root", str(TEMP),
                "--shared-semantic-head", "true",
                "--caption-target-mode", "multi_positive",
                "--image-soft-clip-weight", "1.0",
                "--image-mse-weight", "1000.0",
                "--text-soft-clip-weight", "1.0",
                "--text-mse-weight", "250.0",
                "--text-loss-weight", "0.1",
            ],
            TEMP / "retrainS9_for_50samples.log",
            args.gpu,
        )

        write_status("generating_s9")
        run(
            [
                PYTHON,
                "models/generate_semantic_baseline.py",
                "--checkpoint", str(s9_temp / "best_model.pt"),
                "--run-dir", str(s9),
                "--samples-per-split", "50",
                "--steps", "30",
                "--device", "cuda",
            ],
            s9 / "generation50.log",
            args.gpu,
        )
        validate_gallery(s9)

        write_status("evaluating_s9")
        run(
            [PYTHON, "models/evaluate_semantic_gallery.py", "--run-dir", str(s9), "--device", "cuda"],
            s9 / "evaluation50.log",
            args.gpu,
        )

        shutil.rmtree(TEMP)
        write_status("complete")
    except Exception as error:
        write_status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
