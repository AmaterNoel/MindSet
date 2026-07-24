from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import subprocess
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "dashboard_app"
CACHE_ROOT = PROJECT_ROOT / "dashboard_cache"
REMOTE_ROOT = "/data0/home/longnuoer/Projects/MindKeyAnimator"
SSH_HOST = "server"
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".sh", ".ps1", ".css", ".js", ".ts", ".tsx", ".html",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
PREFETCH_POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="artifact-cache")
PREFETCHED: set[str] = set()
PREFETCH_LOCK = threading.Lock()


def ssh(*args: str, binary: bool = False, timeout: int = 20) -> str | bytes:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", SSH_HOST, *args],
        check=True,
        capture_output=True,
        timeout=timeout,
        text=not binary,
    )
    return result.stdout


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Invalid relative path")
    return str(path)


def gpu_status() -> list[dict[str, Any]]:
    fields = "index,name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw"
    raw = str(ssh("nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"))
    rows = []
    for line in raw.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 8:
            continue
        total, used = float(parts[3]), float(parts[4])
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "memoryTotal": total,
                "memoryUsed": used,
                "memoryPercent": round(used / max(total, 1) * 100, 1),
                "utilization": float(parts[5]),
                "temperature": float(parts[6]),
                "power": float(parts[7]),
            }
        )
    return rows


def project_status() -> dict[str, Any]:
    command = (
        f"cd {REMOTE_ROOT} && "
        "printf 'branch=' && git branch --show-current && "
        "printf 'commit=' && git rev-parse --short HEAD && "
        "printf 'status=' && test -z \"$(git status --porcelain)\" && echo clean || echo modified"
    )
    values = {}
    for line in str(ssh(command)).splitlines():
        key, _, value = line.partition("=")
        values[key] = value
    return values


def source_files() -> list[str]:
    raw = str(ssh(f"cd {REMOTE_ROOT} && git ls-files"))
    return [line for line in raw.splitlines() if line]


def read_source(path: str) -> str:
    relative = safe_relative(path)
    if PurePosixPath(relative).suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError("Unsupported text file")
    return str(ssh(f"cd {REMOTE_ROOT} && git show HEAD:./{relative}"))


def experiment_files() -> list[str]:
    command = (
        f"cd {REMOTE_ROOT} && "
        "find output -maxdepth 6 -type f "
        "\\( -name metrics.jsonl -o -name config.json -o -name test_metrics.json "
        "-o -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.webp' \\) "
        "-printf '%T@|%p\\n' 2>/dev/null | sort -nr | head -1000"
    )
    rows = []
    for line in str(ssh(command, timeout=30)).splitlines():
        _, _, path = line.partition("|")
        if path:
            rows.append(path)
    return rows


def read_output(path: str, binary: bool = False) -> str | bytes:
    relative = safe_relative(path)
    if not relative.startswith("output/"):
        raise ValueError("Only output artifacts are readable")
    suffix = PurePosixPath(relative).suffix.lower()
    allowed = IMAGE_SUFFIXES if binary else {".json", ".jsonl", ".txt", ".log"}
    if suffix not in allowed:
        raise ValueError("Unsupported artifact")
    return ssh(f"cd {REMOTE_ROOT} && cat -- {relative}", binary=binary, timeout=30)


def image_cache_path(path: str) -> Path:
    relative = safe_relative(path)
    if not relative.startswith("output/") or PurePosixPath(relative).suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("Unsupported image artifact")
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
    return CACHE_ROOT / f"{digest}_{PurePosixPath(relative).name}"


def cached_image(path: str) -> bytes:
    destination = image_cache_path(path)
    if destination.exists():
        return destination.read_bytes()
    body = read_output(path, binary=True)
    assert isinstance(body, bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_bytes(body)
    os.replace(temporary, destination)
    return body


def prefetch_images(paths: list[str]) -> None:
    for path in paths:
        if not path.endswith("_comparison.png"):
            continue
        with PREFETCH_LOCK:
            if path in PREFETCHED:
                continue
            PREFETCHED.add(path)
        PREFETCH_POOL.submit(cached_image, path)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "MindSetDashboard/1.0"

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/api/status":
                self.send_json({"gpus": gpu_status(), "project": project_status()})
            elif parsed.path == "/api/tree":
                self.send_json({"files": source_files()})
            elif parsed.path == "/api/file":
                path = query.get("path", [""])[0]
                self.send_json({"path": path, "content": read_source(path)})
            elif parsed.path == "/api/experiments":
                files = experiment_files()
                prefetch_images(files)
                self.send_json({"files": files})
            elif parsed.path == "/api/artifact":
                path = query.get("path", [""])[0]
                body = read_output(path, binary=False)
                self.send_json({"path": path, "content": body})
            elif parsed.path == "/api/image":
                path = query.get("path", [""])[0]
                body = cached_image(path)
                content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.serve_static(parsed.path)
        except (ValueError, subprocess.SubprocessError, OSError) as exc:
            self.send_json({"error": str(exc)}, status=400)

    def serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path == "/" else request_path.lstrip("/")
        path = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in path.parents and path != WEB_ROOT.resolve():
            self.send_error(404)
            return
        if not path.is_file():
            path = WEB_ROOT / "index.html"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the read-only MindSet server dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Refusing non-loopback bind: this dashboard is local-only.")
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"MindSet dashboard: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
