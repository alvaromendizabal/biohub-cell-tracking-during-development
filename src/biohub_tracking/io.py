"""Atomic small-file writes, hashes and UTC receipts; no external services."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterator
import hashlib
import json
import os
import tempfile
import time
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_json(path: Path, value: dict | list) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Invalid relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or ":" in value or p.as_posix() in ("", "."):
        raise ValueError("Unsafe relative path")
    return p.as_posix()


def project_root(start: Path | None = None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for candidate in (p, *p.parents):
        if (candidate / "configs/project.json").is_file() and (candidate / "src/biohub_tracking").is_dir():
            return candidate
    raise RuntimeError("Open the notebook from inside biohub-cell-tracking-during-development.")


@contextmanager
def stage(name: str, log_path: Path) -> Iterator[None]:
    """Records bounded-stage entry/exit. A stage itself is not a timeout mechanism."""
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def emit(state: str) -> None:
        item = {"utc": utc_now(), "stage": name, "state": state,
                "stage_elapsed_seconds": round(time.monotonic() - started, 3)}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item) + "\n")
        print(json.dumps(item), flush=True)
    emit("started")
    try:
        yield
    except BaseException:
        emit("failed_or_interrupted")
        raise
    else:
        emit("completed")
