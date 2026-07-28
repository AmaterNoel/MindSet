from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data0/home/longnuoer/miniconda3/envs/lne3.12/bin/python"
OUTPUT = ROOT / "output" / "fusion_retrieval_50"
TEMP = ROOT / "output_smoke" / "fusion_retrieval_50"
STATUS = OUTPUT / "_status.json"
LOG = OUTPUT / "_suite.log"


def status(stage: str, **extra: object) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(
        json.dumps(
            {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds"), **extra},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run(command: list[str], gpu: int = 5) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    with LOG.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    s7 = TEMP / "S7_retrain"
    g = TEMP / "G_retrain"
    TEMP.mkdir(parents=True, exist_ok=True)
    try:
        if not (s7 / "best_model.pt").exists():
            status("training_s7")
            run(
            [
                PYTHON,
                "models/train_base_model_1D.py",
                "--epochs", "30",
                "--eval-every", "1",
                "--batch-size", "256",
                "--run-name", s7.name,
                "--output-root", str(TEMP),
                "--shared-semantic-head", "true",
                "--caption-target-mode", "min_loss",
                "--image-soft-clip-weight", "0.0",
                "--image-mse-weight", "0.0",
                "--text-soft-clip-weight", "1.0",
                "--text-mse-weight", "1000.0",
                "--text-loss-weight", "1.0",
            ]
            )

        if not (g / "best_low_model.pt").exists():
            status("training_g")
            run(
            [
                PYTHON,
                "models/train_low_model_structured.py",
                "--epochs", "30",
                "--eval-every", "5",
                "--batch-size", "64",
                "--run-name", g.name,
                "--output-root", str(TEMP),
                "--hidden-dim", "2048",
                "--n-blocks", "2",
                "--seed-channels", "64",
                "--multiscale-weight", "1.0",
                "--gradient-weight", "0.1",
                "--ssim-weight", "0.1",
                "--semantic-guidance", "false",
                "--reliability-gate", "false",
                "--fixed-samples-per-split", "5",
                "--delete-checkpoints-after-run", "false",
            ]
            )

        status("generating", samples=50)
        run(
            [
                PYTHON,
                "models/generate_s7_retrieval_img2img.py",
                "--s7-checkpoint", str(s7 / "best_model.pt"),
                "--g-checkpoint", str(g / "best_low_model.pt"),
                "--output-root", str(OUTPUT),
                "--samples", "50",
                "--top-k", "3",
                "--steps", "30",
                "--controlnet-scale", "0.25",
                "--control-guidance-end", "0.4",
                "--device", "cuda",
            ]
        )
        comparison = json.loads((OUTPUT / "comparison.json").read_text(encoding="utf-8"))
        shutil.rmtree(TEMP)
        status("complete", comparison=comparison, checkpoints_deleted=True)
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
