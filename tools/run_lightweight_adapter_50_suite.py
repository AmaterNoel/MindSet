from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data0/home/longnuoer/miniconda3/envs/lne3.12/bin/python"
OUTPUT = ROOT / "output" / "lightweight_adapter_50"
TEMP = ROOT / "output_smoke" / "lightweight_adapter_50"
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


def available_gpu(min_free_mib: int = 30000) -> int | None:
    raw = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    candidates = []
    for line in raw.splitlines():
        index, free, utilization = [int(part.strip()) for part in line.split(",")]
        if index in {1, 5} and free >= min_free_mib:
            candidates.append((utilization, -free, index))
    return min(candidates)[2] if candidates else None


def wait_gpu() -> int:
    while True:
        gpu = available_gpu()
        if gpu is not None:
            return gpu
        status("waiting_for_shared_gpu", allowed_gpus=[1, 5], minimum_free_mib=30000)
        time.sleep(300)


def run(command: list[str], gpu: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    with LOG.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True
        )


def main() -> None:
    s7 = TEMP / "S7"
    g = TEMP / "G"
    adapter = TEMP / "adapters" / "best_adapters.pt"
    TEMP.mkdir(parents=True, exist_ok=True)
    try:
        gpu = wait_gpu()
        if not (s7 / "best_model.pt").exists():
            status("training_S7", gpu=gpu)
            run(
            [
                PYTHON, "models/train_base_model_1D.py",
                "--epochs", "30", "--eval-every", "1", "--batch-size", "256",
                "--run-name", s7.name, "--output-root", str(TEMP),
                "--shared-semantic-head", "true", "--caption-target-mode", "min_loss",
                "--image-soft-clip-weight", "0.0", "--image-mse-weight", "0.0",
                "--text-soft-clip-weight", "1.0", "--text-mse-weight", "1000.0",
                "--text-loss-weight", "1.0",
            ],
                gpu,
            )
        gpu = wait_gpu()
        if not (g / "best_low_model.pt").exists():
            status("training_G", gpu=gpu)
            run(
            [
                PYTHON, "models/train_low_model_structured.py",
                "--epochs", "30", "--eval-every", "5", "--batch-size", "64",
                "--run-name", g.name, "--output-root", str(TEMP),
                "--hidden-dim", "2048", "--n-blocks", "2", "--seed-channels", "64",
                "--multiscale-weight", "1.0", "--gradient-weight", "0.1",
                "--ssim-weight", "0.1", "--semantic-guidance", "false",
                "--reliability-gate", "false", "--fixed-samples-per-split", "5",
                "--delete-checkpoints-after-run", "false",
            ],
                gpu,
            )
        gpu = wait_gpu()
        if not adapter.exists():
            status("training_adapters", gpu=gpu)
            run(
            [
                PYTHON, "models/train_lightweight_fusion_adapters.py",
                "--s7-checkpoint", str(s7 / "best_model.pt"),
                "--g-checkpoint", str(g / "best_low_model.pt"),
                "--output", str(adapter), "--epochs", "20",
                "--batch-size", "128", "--device", "cuda",
            ],
                gpu,
            )
        gpu = wait_gpu()
        status("generating_and_public_evaluation", gpu=gpu, samples=50)
        run(
            [
                PYTHON, "models/generate_lightweight_adapter_suite.py",
                "--s7-checkpoint", str(s7 / "best_model.pt"),
                "--g-checkpoint", str(g / "best_low_model.pt"),
                "--adapter-checkpoint", str(adapter),
                "--output-root", str(OUTPUT), "--samples", "50",
                "--steps", "35", "--device", "cuda",
            ],
            gpu,
        )
        comparison = json.loads((OUTPUT / "comparison.json").read_text(encoding="utf-8"))
        shutil.rmtree(TEMP)
        status("complete", comparison=comparison, checkpoints_deleted=True)
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
