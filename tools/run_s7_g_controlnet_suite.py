from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data0/home/longnuoer/miniconda3/envs/lne3.12/bin/python"


def write_status(path: Path, stage: str, **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds"), **extra},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run(command: list[str], log_path: Path, gpu: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain S7/G and run full-test S7+G ControlNet comparison.")
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--controlnet-scale", type=float, default=0.65)
    args = parser.parse_args()

    output_root = ROOT / "output" / "fusion_model"
    temp_root = ROOT / "output_smoke" / "fusion_s7_g"
    status = output_root / "_status.json"
    log = output_root / "_suite.log"
    s7_temp = temp_root / "S7_retrain"
    g_temp = temp_root / "G_retrain"
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        write_status(status, "training_s7")
        run(
            [
                PYTHON,
                "models/train_base_model_1D.py",
                "--epochs",
                "30",
                "--eval-every",
                "1",
                "--batch-size",
                "256",
                "--run-name",
                s7_temp.name,
                "--output-root",
                str(temp_root),
                "--shared-semantic-head",
                "true",
                "--caption-target-mode",
                "min_loss",
                "--image-soft-clip-weight",
                "0.0",
                "--image-mse-weight",
                "0.0",
                "--text-soft-clip-weight",
                "1.0",
                "--text-mse-weight",
                "1000.0",
                "--text-loss-weight",
                "1.0",
            ],
            log,
            args.gpu,
        )

        write_status(status, "training_g")
        run(
            [
                PYTHON,
                "models/train_low_model_structured.py",
                "--epochs",
                "30",
                "--eval-every",
                "5",
                "--batch-size",
                "64",
                "--run-name",
                g_temp.name,
                "--output-root",
                str(temp_root),
                "--hidden-dim",
                "2048",
                "--n-blocks",
                "2",
                "--seed-channels",
                "64",
                "--multiscale-weight",
                "1.0",
                "--gradient-weight",
                "0.1",
                "--ssim-weight",
                "0.1",
                "--semantic-guidance",
                "false",
                "--reliability-gate",
                "false",
                "--fixed-samples-per-split",
                "5",
                "--delete-checkpoints-after-run",
                "false",
            ],
            log,
            args.gpu,
        )

        write_status(status, "generating_and_evaluating", expected_test_samples=354)
        run(
            [
                PYTHON,
                "models/generate_s7_g_controlnet.py",
                "--s7-checkpoint",
                str(s7_temp / "best_model.pt"),
                "--g-checkpoint",
                str(g_temp / "best_low_model.pt"),
                "--output-root",
                str(output_root),
                "--controlnet-scale",
                str(args.controlnet_scale),
                "--device",
                "cuda",
            ],
            log,
            args.gpu,
        )

        comparison = json.loads((output_root / "comparison.json").read_text(encoding="utf-8"))
        shutil.rmtree(temp_root)
        write_status(status, "complete", comparison=comparison, checkpoints_deleted=True)
    except Exception as error:
        write_status(status, "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
