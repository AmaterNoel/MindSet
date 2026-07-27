from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data0/home/longnuoer/miniconda3/envs/lne3.12/bin/python")


@dataclass(frozen=True)
class Experiment:
    name: str
    caption_mode: str
    shared_head: bool
    image_soft_weight: float
    image_mse_weight: float
    text_soft_weight: float
    text_mse_weight: float
    text_loss_weight: float


EXPERIMENTS = [
    Experiment("expS3_image_only_30ep", "mean", True, 1.0, 1000.0, 1.0, 1000.0, 0.0),
    Experiment("expS4_caption_mean_only_30ep", "mean", True, 0.0, 0.0, 1.0, 1000.0, 1.0),
    Experiment("expS5_caption_best_only_30ep", "best", True, 0.0, 0.0, 1.0, 1000.0, 1.0),
    Experiment("expS6_caption_softbest_only_30ep", "soft_best", True, 0.0, 0.0, 1.0, 1000.0, 1.0),
    Experiment("expS7_caption_minloss_only_30ep", "min_loss", True, 0.0, 0.0, 1.0, 1000.0, 1.0),
    Experiment("expS8_caption_multipos_only_30ep", "multi_positive", True, 0.0, 0.0, 1.0, 250.0, 1.0),
    Experiment("expS9_image_weak_multipos_30ep", "multi_positive", True, 1.0, 1000.0, 1.0, 250.0, 0.1),
    Experiment("expS10_dualhead_softbest_30ep", "soft_best", False, 1.0, 1000.0, 1.0, 1000.0, 1.0),
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str, path: Path) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def gpu_snapshot() -> list[dict[str, Any]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    apps_raw = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    active_uuids = {line.strip() for line in apps_raw.splitlines() if line.strip()}
    rows = []
    for line in query.splitlines():
        index, uuid, free, utilization = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "free": int(free),
                "utilization": int(utilization),
                "has_compute_process": uuid in active_uuids,
            }
        )
    return rows


def wait_for_idle_gpu(min_free_mb: int, poll_seconds: int, log_path: Path) -> int:
    while True:
        first = gpu_snapshot()
        candidates = [
            gpu
            for gpu in first
            if not gpu["has_compute_process"]
            and gpu["free"] >= min_free_mb
            and gpu["utilization"] <= 10
        ]
        if candidates:
            time.sleep(10)
            second = {gpu["index"]: gpu for gpu in gpu_snapshot()}
            stable = [
                gpu
                for gpu in candidates
                if gpu["index"] in second
                and not second[gpu["index"]]["has_compute_process"]
                and second[gpu["index"]]["free"] >= min_free_mb
                and second[gpu["index"]]["utilization"] <= 10
            ]
            if stable:
                chosen = max(stable, key=lambda gpu: second[gpu["index"]]["free"])
                log(
                    f"selected GPU {chosen['index']} "
                    f"(free={second[chosen['index']]['free']} MiB, util={second[chosen['index']]['utilization']}%)",
                    log_path,
                )
                return int(chosen["index"])
        summary = ", ".join(
            f"{gpu['index']}:free={gpu['free']} util={gpu['utilization']} active={gpu['has_compute_process']}"
            for gpu in first
        )
        log(f"no stable idle GPU; next check in {poll_seconds}s; {summary}", log_path)
        time.sleep(poll_seconds)


def run_stage(command: list[str], gpu: int, log_path: Path, suite_log: Path, label: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    log(f"{label} started on GPU {gpu}", suite_log)
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}; see {log_path}")
    log(f"{label} finished on GPU {gpu}", suite_log)


def run_stage_with_gpu_retries(
    command_factory: Any,
    min_free_mb: int,
    poll_seconds: int,
    log_path: Path,
    suite_log: Path,
    label: str,
    before_retry: Any | None = None,
    max_attempts: int = 3,
) -> int:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        gpu = wait_for_idle_gpu(min_free_mb, poll_seconds, suite_log)
        try:
            run_stage(command_factory(), gpu, log_path, suite_log, f"{label} attempt {attempt}")
            return gpu
        except Exception as error:
            last_error = error
            log(f"{label} attempt {attempt} failed: {error}", suite_log)
            if before_retry is not None:
                before_retry()
            if attempt < max_attempts:
                log(f"{label} will return to idle-GPU wait", suite_log)
    assert last_error is not None
    raise last_error


def write_status(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def train_command(experiment: Experiment, output_root: Path) -> list[str]:
    return [
        str(PYTHON),
        "models/train_base_model_1D.py",
        "--epochs",
        "30",
        "--eval-every",
        "1",
        "--batch-size",
        "256",
        "--run-name",
        experiment.name,
        "--output-root",
        str(output_root),
        "--shared-semantic-head",
        str(experiment.shared_head).lower(),
        "--caption-target-mode",
        experiment.caption_mode,
        "--image-soft-clip-weight",
        str(experiment.image_soft_weight),
        "--image-mse-weight",
        str(experiment.image_mse_weight),
        "--text-soft-clip-weight",
        str(experiment.text_soft_weight),
        "--text-mse-weight",
        str(experiment.text_mse_weight),
        "--text-loss-weight",
        str(experiment.text_loss_weight),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-shot S3-S10 semantic experiment suite.")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output" / "semantic_model")
    args = parser.parse_args()

    suite_dir = args.output_root / "_suite_s3_s10"
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_log = suite_dir / "suite.log"
    status_path = suite_dir / "status.json"
    status: dict[str, Any] = {
        "started_at": now(),
        "state": "running",
        "experiments": {experiment.name: "pending" for experiment in EXPERIMENTS},
    }
    write_status(status_path, status)

    try:
        for experiment in EXPERIMENTS:
            run_dir = args.output_root / experiment.name
            run_dir.mkdir(parents=True, exist_ok=True)
            if (run_dir / "generation_metrics.json").exists() and not list(run_dir.glob("*.pt")):
                status["experiments"][experiment.name] = "complete"
                write_status(status_path, status)
                log(f"{experiment.name} already complete; skipped", suite_log)
                continue

            status["current"] = experiment.name
            status["experiments"][experiment.name] = "waiting_for_training_gpu"
            write_status(status_path, status)
            def reset_partial_training() -> None:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)

            training_gpu = run_stage_with_gpu_retries(
                lambda: train_command(experiment, args.output_root),
                4096,
                args.poll_seconds,
                run_dir / "train.log",
                suite_log,
                f"{experiment.name} training",
                before_retry=reset_partial_training,
            )
            status["experiments"][experiment.name] = f"trained_gpu_{training_gpu}"
            write_status(status_path, status)

            status["experiments"][experiment.name] = "waiting_for_generation_gpu"
            write_status(status_path, status)
            generation_gpu = run_stage_with_gpu_retries(
                lambda: [
                    str(PYTHON),
                    "models/generate_semantic_baseline.py",
                    "--checkpoint",
                    str(run_dir / "best_model.pt"),
                    "--run-dir",
                    str(run_dir),
                    "--samples-per-split",
                    "5",
                    "--steps",
                    "30",
                    "--device",
                    "cuda",
                ],
                12288,
                args.poll_seconds,
                run_dir / "generation.log",
                suite_log,
                f"{experiment.name} generation",
            )
            status["experiments"][experiment.name] = f"generated_gpu_{generation_gpu}"
            write_status(status_path, status)
            run_stage_with_gpu_retries(
                lambda: [
                    str(PYTHON),
                    "models/evaluate_semantic_gallery.py",
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    "cuda",
                ],
                4096,
                args.poll_seconds,
                run_dir / "evaluation.log",
                suite_log,
                f"{experiment.name} evaluation",
            )
            for checkpoint in (run_dir / "best_model.pt", run_dir / "last_model.pt"):
                checkpoint.unlink(missing_ok=True)
            status["experiments"][experiment.name] = "complete"
            write_status(status_path, status)
            log(f"{experiment.name} complete; checkpoints removed", suite_log)

        status["state"] = "complete"
        status["finished_at"] = now()
        status.pop("current", None)
        write_status(status_path, status)
        log("S3-S10 suite complete", suite_log)
    except Exception as error:
        status["state"] = "failed"
        status["error"] = str(error)
        status["failed_at"] = now()
        write_status(status_path, status)
        log(f"suite failed: {error}", suite_log)
        raise


if __name__ == "__main__":
    if os.name == "nt":
        raise SystemExit("This queue runner is intended for the Linux training server.")
    sys.exit(main())
