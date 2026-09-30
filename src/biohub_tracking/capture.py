"""One bounded, user-triggered, read-only Kaggle file-list request; no data download."""
from __future__ import annotations
from importlib import metadata
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time
from .io import atomic_json, atomic_text, run_id, sha256_file, stage, utc_now
from .inventory import parse_cli_csv

COMPETITION = "biohub-cell-tracking-during-development"


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=5)


def capture_first_page(root: Path, seconds: int = 90) -> tuple[dict, list[dict]]:
    """Reuse a compatible, hash-verified local snapshot; otherwise make one request."""
    root = Path(root).resolve()
    if not 1 <= seconds <= 90:
        raise ValueError("This milestone permits a timeout from 1 to 90 seconds only")
    signature = {"competition": COMPETITION, "page_size": 200,
                 "kaggle_version": metadata.version("kaggle"),
                 "capture_code_sha256": sha256_file(Path(__file__)),
                 "parser_code_sha256": sha256_file(Path(__file__).with_name("inventory.py"))}
    latest = root / "outputs/inventory/latest.json"
    if latest.is_file():
        previous = json.loads(latest.read_text())
        if previous.get("signature") == signature and previous.get("status") == "completed":
            raw = root / previous["raw_relative_path"]
            # Only trust the expected local inventory subtree, not arbitrary receipt paths.
            if (root / "outputs/inventory").resolve() not in raw.resolve().parents:
                raise ValueError("Unsafe inventory receipt path")
            if not raw.is_file() or sha256_file(raw) != previous["raw_sha256"]:
                raise ValueError("Prior inventory checksum mismatch; stop and inspect it.")
            rows = parse_cli_csv(raw.read_text(encoding="utf-8"))
            print(f"Reusing validated local metadata snapshot from {previous['utc']}; not a fresh listing.")
            return previous, rows
    folder = root / "outputs/inventory" / run_id()
    folder.mkdir(parents=True, exist_ok=False)
    raw = folder / "kaggle_files_first_page.csv"
    errors = folder / "stderr_private.txt"
    exe = Path(sys.executable).parent / "kaggle"
    if not exe.is_file():
        raise RuntimeError("Kaggle CLI not found in the active project environment")
    command = [str(exe), "competitions", "files", COMPETITION, "--page-size", "200", "--csv"]
    started = time.monotonic()
    proc = None
    try:
        with stage("kaggle_first_metadata_page", root / "logs/stages.jsonl"):
            with raw.open("w", encoding="utf-8") as out, errors.open("w", encoding="utf-8") as err:
                proc = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                        stdout=out, stderr=err, start_new_session=(os.name == "posix"))
                last = started
                while proc.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed > seconds:
                        raise TimeoutError("Kaggle metadata request exceeded its 90-second maximum")
                    if raw.stat().st_size + errors.stat().st_size > 2 * 1024**2:
                        raise RuntimeError("Metadata response exceeded its 2 MiB safety bound")
                    if time.monotonic() - last >= 10:
                        print(json.dumps({"utc": utc_now(), "stage": "kaggle_first_metadata_page",
                                          "elapsed_seconds": round(elapsed, 1),
                                          "stdout_bytes_observed": raw.stat().st_size,
                                          "note": "Waiting for API; this is not a completed file count."}), flush=True)
                        last = time.monotonic()
                    time.sleep(0.2)
            if proc.returncode != 0:
                raise RuntimeError(f"Kaggle CLI exit {proc.returncode}; inspect stderr_private.txt locally. "
                                   "Check competition entry/authentication before retrying. Do not share credentials.")
            rows = parse_cli_csv(raw.read_text(encoding="utf-8"))
            receipt = {"utc": utc_now(), "status": "completed", "signature": signature,
                       "scope": "FIRST_PAGE_ONLY_NOT_COMPLETE_INVENTORY", "inventory_complete": False,
                       "listed_file_count": len(rows), "data_files_downloaded": 0,
                       "raw_relative_path": raw.relative_to(root).as_posix(),
                       "raw_sha256": sha256_file(raw),
                       "elapsed_seconds": round(time.monotonic() - started, 3)}
            atomic_json(folder / "receipt.json", receipt)
            atomic_json(latest, receipt)
            return receipt, rows
    except BaseException as exc:
        if proc is not None:
            _terminate(proc)
        atomic_json(folder / "failure.json", {"utc": utc_now(), "status": "failed_or_interrupted",
                    "error_type": type(exc).__name__, "signature": signature,
                    "elapsed_seconds": round(time.monotonic() - started, 3)})
        raise
