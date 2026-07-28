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
OUTPUT = ROOT / "output" / "embedding_spatial_oracle_50"
TEMP = ROOT / "output_smoke" / "embedding_spatial_oracle_50"
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


def idle_gpu(min_free_mib: int = 26000) -> int | None:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    active = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    active_uuids = {line.strip() for line in active.splitlines() if line.strip()}
    candidates = []
    for line in query.splitlines():
        index, uuid, free = [part.strip() for part in line.split(",")]
        if uuid not in active_uuids and int(free) >= min_free_mib:
            candidates.append((int(free), int(index)))
    return max(candidates)[1] if candidates else None


def wait_for_gpu() -> int:
    while True:
        gpu = idle_gpu()
        if gpu is not None:
            return gpu
        status("waiting_for_idle_gpu", poll_seconds=300)
        time.sleep(300)


def run(command: list[str], gpu: int) -> None:
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
    semantic = TEMP / "image_embedding_model"
    low = TEMP / "G_low_model"
    TEMP.mkdir(parents=True, exist_ok=True)
    try:
        gpu = wait_for_gpu()
        if not (semantic / "best_model.pt").exists():
            status("training_direct_image_embedding", gpu=gpu)
            run(
                [
                    PYTHON,
                    "models/train_base_model_1D.py",
                    "--epochs", "30",
                    "--eval-every", "1",
                    "--batch-size", "256",
                    "--run-name", semantic.name,
                    "--output-root", str(TEMP),
                    "--shared-semantic-head", "true",
                    "--caption-target-mode", "mean",
                    "--image-soft-clip-weight", "1.0",
                    "--image-mse-weight", "1000.0",
                    "--text-soft-clip-weight", "0.0",
                    "--text-mse-weight", "0.0",
                    "--text-loss-weight", "0.0",
                ],
                gpu,
            )
        if not (low / "best_low_model.pt").exists():
            gpu = wait_for_gpu()
            status("training_G", gpu=gpu)
            run(
                [
                    PYTHON,
                    "models/train_low_model_structured.py",
                    "--epochs", "30",
                    "--eval-every", "5",
                    "--batch-size", "64",
                    "--run-name", low.name,
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
                ],
                gpu,
            )
        gpu = wait_for_gpu()
        status("generating_three_experiments", gpu=gpu, samples=50)
        run(
            [
                PYTHON,
                "models/generate_embedding_spatial_oracle.py",
                "--s7-checkpoint", str(semantic / "best_model.pt"),
                "--g-checkpoint", str(low / "best_low_model.pt"),
                "--output-root", str(OUTPUT),
                "--samples", "50",
                "--steps", "35",
                "--ip-adapter-scale", "1.0",
                "--controlnet-scale", "0.8",
                "--control-guidance-end", "0.9",
                "--device", "cuda",
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
