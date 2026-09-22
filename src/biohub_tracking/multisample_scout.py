from __future__ import annotations

import importlib.util
import itertools
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd

from .feature_io import atomic, write_json
from .geff_snapshot import resolve_kaggle_command


ARRAY_PATHS = [
    "nodes/ids",
    "nodes/props/t/values",
    "nodes/props/z/values",
    "nodes/props/y/values",
    "nodes/props/x/values",
    "edges/ids",
]


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_recursive(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_recursive(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_recursive(value, key)
            if found is not None:
                return found
    return None


def verify_google_crc32c() -> dict:
    google = importlib.util.find_spec("google_crc32c") is not None
    legacy = importlib.util.find_spec("crc32c") is not None
    if not google:
        raise RuntimeError(
            "google-crc32c is not installed. Install google-crc32c before "
            "continuing so numcodecs does not rely on the deprecated crc32c backend."
        )
    return {
        "google_crc32c_available": google,
        "legacy_crc32c_available": legacy,
        "backend_policy": "google_crc32c_preferred",
    }



_LAST_NETWORK_REQUEST_MONOTONIC = None
_CUMULATIVE_RETRY_SLEEP_SECONDS = 0.0


def _project_root_from_destination(destination_root: Path) -> Path:
    destination_root = Path(destination_root).resolve()
    if destination_root.name != "multisample_scout":
        raise ValueError(
            f"Unexpected scout destination root: {destination_root}"
        )
    return destination_root.parents[1]


def _sanitize_network_text(text: str) -> str:
    """Remove signed URL query strings/tokens from subprocess diagnostics."""
    text = str(text or "")
    import re

    def repl(match):
        raw = match.group(0)
        trailing = ""
        while raw and raw[-1] in ".,);]":
            trailing = raw[-1] + trailing
            raw = raw[:-1]
        try:
            parts = urlsplit(raw)
            clean = urlunsplit(
                (parts.scheme, parts.netloc, parts.path, "", "")
            )
        except Exception:
            clean = "[REDACTED_URL]"
        return clean + trailing

    return re.sub(r"https?://[^\s]+", repl, text)


def _error_class(text: str) -> str:
    lowered = str(text or "").lower()
    if "429" in lowered or "too many requests" in lowered:
        return "rate_limit"
    for status in ("500", "502", "503", "504"):
        if (
            f" {status} " in f" {lowered} "
            or f"{status} server error" in lowered
            or f"http {status}" in lowered
        ):
            return "transient_server"
    return "fatal"


def _append_download_event(
    project_root: Path,
    *,
    remote: str,
    attempt: int,
    outcome: str,
    error_class: str | None = None,
    detail: str | None = None,
    sleep_seconds: float | None = None,
) -> None:
    out = project_root / "outputs/multisample_scout"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "download_events.jsonl"
    payload = {
        "utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).replace(microsecond=0).isoformat(),
        "remote": str(remote),
        "attempt": int(attempt),
        "outcome": str(outcome),
        "error_class": error_class,
        "detail": _sanitize_network_text(detail or "")[-1600:],
        "sleep_seconds": (
            None if sleep_seconds is None else float(sleep_seconds)
        ),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _pace_request(seconds: float) -> None:
    global _LAST_NETWORK_REQUEST_MONOTONIC
    seconds = max(0.0, float(seconds))
    now = time.monotonic()
    if _LAST_NETWORK_REQUEST_MONOTONIC is not None:
        elapsed = now - _LAST_NETWORK_REQUEST_MONOTONIC
        remaining = seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
    _LAST_NETWORK_REQUEST_MONOTONIC = time.monotonic()


def reset_rate_limit_budget() -> None:
    global _LAST_NETWORK_REQUEST_MONOTONIC
    global _CUMULATIVE_RETRY_SLEEP_SECONDS
    _LAST_NETWORK_REQUEST_MONOTONIC = None
    _CUMULATIVE_RETRY_SLEEP_SECONDS = 0.0


def _retry_or_stop(
    *,
    project_root: Path,
    remote: str,
    attempt: int,
    detail: str,
    error_class: str,
    retry_delays_seconds,
    max_cumulative_retry_sleep_seconds: float,
) -> bool:
    """Sleep for a bounded retry. Return True to retry, False to stop."""
    global _CUMULATIVE_RETRY_SLEEP_SECONDS

    if error_class not in {"rate_limit", "transient_server"}:
        return False

    retry_index = attempt - 1
    if retry_index >= len(retry_delays_seconds):
        return False

    delay = float(retry_delays_seconds[retry_index])
    if (
        _CUMULATIVE_RETRY_SLEEP_SECONDS + delay
        > float(max_cumulative_retry_sleep_seconds)
    ):
        return False

    _append_download_event(
        project_root,
        remote=remote,
        attempt=attempt,
        outcome="retry_scheduled",
        error_class=error_class,
        detail=detail,
        sleep_seconds=delay,
    )
    time.sleep(delay)
    _CUMULATIVE_RETRY_SLEEP_SECONDS += delay
    return True


def _download_one(
    competition: str,
    remote: str,
    destination_root: Path,
    *,
    timeout_seconds=90,
    request_interval_seconds=4.0,
    retry_delays_seconds=(15.0, 30.0, 60.0),
    max_cumulative_retry_sleep_seconds=105.0,
) -> Path:
    destination_root = Path(destination_root)
    project_root = _project_root_from_destination(destination_root)
    target = destination_root / remote

    if target.is_file() and target.stat().st_size > 0:
        _append_download_event(
            project_root,
            remote=remote,
            attempt=0,
            outcome="cache_reuse",
        )
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    max_attempts = 1 + len(tuple(retry_delays_seconds))

    for attempt in range(1, max_attempts + 1):
        _pace_request(request_interval_seconds)

        cmd = [
            *resolve_kaggle_command(),
            "competitions",
            "download",
            competition,
            "-f",
            remote,
            "-p",
            str(target.parent),
            "-o",
            "-q",
        ]

        p = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )

        if p.returncode == 0:
            if target.is_file() and target.stat().st_size > 0:
                _append_download_event(
                    project_root,
                    remote=remote,
                    attempt=attempt,
                    outcome="download_success",
                )
                return target

            zip_candidate = target.with_name(target.name + ".zip")
            if zip_candidate.is_file():
                import zipfile
                with zipfile.ZipFile(zip_candidate) as z:
                    members = [
                        m for m in z.infolist()
                        if not m.is_dir()
                        and Path(m.filename).name == target.name
                        and ".." not in Path(m.filename).parts
                    ]
                    if len(members) != 1:
                        raise FileNotFoundError(
                            f"Could not uniquely identify {target.name} "
                            "inside Kaggle ZIP wrapper."
                        )
                    with z.open(members[0]) as src_f, target.open("wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)
                zip_candidate.unlink()
                if target.is_file() and target.stat().st_size > 0:
                    _append_download_event(
                        project_root,
                        remote=remote,
                        attempt=attempt,
                        outcome="download_success_zip_wrapper",
                    )
                    return target

            _append_download_event(
                project_root,
                remote=remote,
                attempt=attempt,
                outcome="success_missing_target",
                error_class="fatal",
                detail=(
                    "Kaggle command returned success but target was absent."
                ),
            )
            raise FileNotFoundError(
                f"Kaggle returned success but requested file is missing: {target}"
            )

        raw = (p.stderr or p.stdout or "")[-4000:]
        sanitized = _sanitize_network_text(raw)
        cls = _error_class(sanitized)

        _append_download_event(
            project_root,
            remote=remote,
            attempt=attempt,
            outcome="download_error",
            error_class=cls,
            detail=sanitized,
        )

        if _retry_or_stop(
            project_root=project_root,
            remote=remote,
            attempt=attempt,
            detail=sanitized,
            error_class=cls,
            retry_delays_seconds=tuple(retry_delays_seconds),
            max_cumulative_retry_sleep_seconds=(
                max_cumulative_retry_sleep_seconds
            ),
        ):
            continue

        if cls == "rate_limit":
            state = {
                "status": "rate_limited",
                "remote": remote,
                "attempts": attempt,
                "max_attempts": max_attempts,
                "cumulative_retry_sleep_seconds": (
                    _CUMULATIVE_RETRY_SLEEP_SECONDS
                ),
                "request_interval_seconds": float(request_interval_seconds),
                "last_error": sanitized[-1600:],
                "signed_url_query_logged": False,
            }
            write_json(
                project_root
                / "outputs/multisample_scout/rate_limit_state.json",
                state,
            )
            raise RuntimeError(
                "KAGGLE_RATE_LIMIT_STOP "
                f"remote={remote!r} attempts={attempt}; "
                "bounded retry budget exhausted. "
                "Do not rerun unchanged."
            )

        raise RuntimeError(
            f"Kaggle download failed for {remote!r}: {sanitized[-1600:]}"
        )

    raise RuntimeError(
        f"Kaggle download loop exited unexpectedly for {remote!r}"
    )

def _chunk_layout(meta: dict):
    shape = tuple(int(v) for v in meta["shape"])
    grid = meta.get("chunk_grid", {}).get("configuration", {})
    chunk_shape = tuple(int(v) for v in grid["chunk_shape"])
    enc = meta.get("chunk_key_encoding", {})
    name = enc.get("name", "default")
    conf = enc.get("configuration", {})
    separator = conf.get("separator", "/")
    if name not in ("default", "v2"):
        raise ValueError(f"Unsupported chunk key encoding: {name}")
    return shape, chunk_shape, name, separator


def _chunk_keys(array_path: str, meta: dict) -> list[str]:
    shape, chunk_shape, name, separator = _chunk_layout(meta)
    counts = [
        int(math.ceil(s / c))
        for s, c in zip(shape, chunk_shape, strict=True)
    ]
    keys = []
    for coord in itertools.product(*[range(n) for n in counts]):
        if name == "default":
            suffix = "c" + separator + separator.join(str(int(v)) for v in coord)
        else:
            suffix = separator.join(str(int(v)) for v in coord)
        keys.append(f"{array_path}/{suffix}")
    return keys


def _decoded_bytes(meta: dict) -> int:
    dtype = np.dtype(meta["data_type"])
    return int(np.prod(meta["shape"]) * dtype.itemsize)


def acquire_geff_sample(
    root: Path,
    sample_id: str,
    *,
    competition: str,
    max_files: int,
    max_decoded_bytes: int,
    request_interval_seconds: float,
    retry_delays_seconds,
    max_cumulative_retry_sleep_seconds: float,
) -> dict:
    root = Path(root)
    store = root / "data/multisample_scout/train" / f"{sample_id}.geff"
    prefix = f"train/{sample_id}.geff/"
    dest_root = root / "data/multisample_scout"

    root_meta = _download_one(
        competition,
        prefix + "zarr.json",
        dest_root,
        request_interval_seconds=request_interval_seconds,
        retry_delays_seconds=retry_delays_seconds,
        max_cumulative_retry_sleep_seconds=max_cumulative_retry_sleep_seconds,
    )
    root_payload = read_json(root_meta)

    remotes = [prefix + "zarr.json"]
    decoded = 0

    for array_path in ARRAY_PATHS:
        meta_remote = prefix + array_path + "/zarr.json"
        meta_path = _download_one(
            competition,
            meta_remote,
            dest_root,
            request_interval_seconds=request_interval_seconds,
            retry_delays_seconds=retry_delays_seconds,
            max_cumulative_retry_sleep_seconds=max_cumulative_retry_sleep_seconds,
        )
        meta = read_json(meta_path)
        remotes.append(meta_remote)
        decoded += _decoded_bytes(meta)
        for key in _chunk_keys(array_path, meta):
            remotes.append(prefix + key)

    unique_remotes = list(dict.fromkeys(remotes))
    if len(unique_remotes) > int(max_files):
        raise RuntimeError(
            f"{sample_id}: planned {len(unique_remotes)} GEFF files "
            f"above cap {max_files}"
        )
    if decoded > int(max_decoded_bytes):
        raise RuntimeError(
            f"{sample_id}: decoded GEFF arrays {decoded} bytes "
            f"above cap {max_decoded_bytes}"
        )

    downloaded = 0
    reused = 0
    for remote in unique_remotes:
        target = dest_root / remote
        existed = target.is_file() and target.stat().st_size > 0
        _download_one(
            competition,
            remote,
            dest_root,
            request_interval_seconds=request_interval_seconds,
            retry_delays_seconds=retry_delays_seconds,
            max_cumulative_retry_sleep_seconds=max_cumulative_retry_sleep_seconds,
        )
        if existed:
            reused += 1
        else:
            downloaded += 1

    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r"crc32c usage is deprecated since numcodecs v0\.16\.4.*",
        category=DeprecationWarning,
        module=r"numcodecs(\..*)?",
    )
    import zarr

    ids = np.asarray(zarr.open_array(store / "nodes/ids", mode="r"))
    t = np.asarray(zarr.open_array(store / "nodes/props/t/values", mode="r"))
    z = np.asarray(zarr.open_array(store / "nodes/props/z/values", mode="r"))
    y = np.asarray(zarr.open_array(store / "nodes/props/y/values", mode="r"))
    x = np.asarray(zarr.open_array(store / "nodes/props/x/values", mode="r"))
    edges = np.asarray(zarr.open_array(store / "edges/ids", mode="r"))

    nodes = pd.DataFrame({
        "gt_id": ids.astype(np.int64),
        "t": t.astype(int),
        "z": z.astype(float),
        "y": y.astype(float),
        "x": x.astype(float),
    })
    edge_df = pd.DataFrame(
        edges.astype(np.int64),
        columns=["source_gt_id", "target_gt_id"],
    )

    sample_out = root / "outputs/multisample_scout/samples" / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)
    atomic(sample_out / "gt_nodes.csv", nodes.to_csv(index=False))
    atomic(sample_out / "gt_edges.csv", edge_df.to_csv(index=False))

    return {
        "sample_id": sample_id,
        "root_meta": root_payload,
        "nodes": nodes,
        "edges": edge_df,
        "planned_files": len(unique_remotes),
        "downloaded_files": downloaded,
        "reused_files": reused,
        "decoded_bytes": decoded,
    }


def negative_link_opportunities(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    pair_radius_um,
    spacing_um,
) -> dict:
    spacing = np.asarray(spacing_um, dtype=float)
    edge_set = set(
        zip(
            edges["source_gt_id"].astype(int),
            edges["target_gt_id"].astype(int),
            strict=True,
        )
    )
    out_sources = set(edges["source_gt_id"].astype(int))
    in_targets = set(edges["target_gt_id"].astype(int))

    positive = set()
    negative = set()
    candidate_pairs = 0

    frames = sorted(int(v) for v in nodes["t"].unique())
    frame_set = set(frames)
    for t in frames:
        if t + 1 not in frame_set:
            continue
        src = nodes[nodes["t"] == t]
        tgt = nodes[nodes["t"] == t + 1]
        if src.empty or tgt.empty:
            continue

        sp = src[["z", "y", "x"]].to_numpy(float)
        tp = tgt[["z", "y", "x"]].to_numpy(float)
        dist = np.linalg.norm(
            (sp[:, None, :] - tp[None, :, :])
            * spacing[None, None, :],
            axis=2,
        )

        for i, srow in enumerate(src.itertuples()):
            sid = int(srow.gt_id)
            for j, trow in enumerate(tgt.itertuples()):
                if float(dist[i, j]) > float(pair_radius_um):
                    continue
                candidate_pairs += 1
                tid = int(trow.gt_id)
                pair = (sid, tid)
                if pair in edge_set:
                    positive.add(pair)
                elif sid in out_sources or tid in in_targets:
                    negative.add(pair)

    out_degree = edges.groupby("source_gt_id").size().rename("out_degree")
    divisions = int((out_degree >= 2).sum())
    max_simultaneous = (
        int(nodes.groupby("t").size().max()) if len(nodes) else 0
    )

    return {
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "frames_with_annotations": int(nodes["t"].nunique()),
        "max_simultaneous_annotated_nodes": max_simultaneous,
        "division_sources": divisions,
        "adjacent_candidate_pairs_15um": int(candidate_pairs),
        "positive_link_opportunities": int(len(positive)),
        "negative_link_opportunities": int(len(negative)),
    }


def run_rate_limit_probe(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/multisample_scout.json")
    verify_google_crc32c()
    reset_rate_limit_budget()

    remote = cfg["rate_limit_probe_remote"]
    destination_root = root / "data/multisample_scout"
    path = _download_one(
        cfg["competition"],
        remote,
        destination_root,
        request_interval_seconds=cfg["request_interval_seconds"],
        retry_delays_seconds=cfg["retry_delays_seconds"],
        max_cumulative_retry_sleep_seconds=cfg[
            "max_cumulative_retry_sleep_seconds"
        ],
    )
    receipt = {
        "status": "passed",
        "sample_id": cfg["rate_limit_probe_sample_id"],
        "remote": remote,
        "path": str(path.relative_to(root)),
        "bytes": int(path.stat().st_size),
        "network_policy": cfg["network_policy"],
        "signed_url_query_logged": False,
    }
    write_json(
        root / "outputs/multisample_scout/rate_limit_probe.json",
        receipt,
    )
    return receipt


def run_scout(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(root / "configs/multisample_scout.json")
    backend = verify_google_crc32c()

    sample_ids = list(dict.fromkeys(cfg["bootstrap_sample_ids"]))
    embryos = sorted({sid.split("_", 1)[0] for sid in sample_ids})
    if len(sample_ids) > cfg["max_samples_to_download"]:
        raise RuntimeError("Bootstrap list exceeds sample cap.")
    if len(embryos) < cfg["target_distinct_embryos"]:
        raise RuntimeError(
            f"Bootstrap covers {len(embryos)} embryos but target is "
            f"{cfg['target_distinct_embryos']}."
        )

    discovery = {
        "sample_ids": sample_ids,
        "per_embryo": {
            embryo: sum(sid.startswith(embryo + "_") for sid in sample_ids)
            for embryo in embryos
        },
        "pages": 0,
        "files_seen": 0,
        "complete": True,
        "discovery_mode": "deterministic_cross_embryo_bootstrap",
    }

    out = root / "outputs/multisample_scout"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "discovery.json", discovery)

    rows = []
    total_cache_bytes = 0
    reset_rate_limit_budget()

    for sample_id in sample_ids:
        acquired = acquire_geff_sample(
            root,
            sample_id,
            competition=cfg["competition"],
            max_files=cfg["max_geff_files_per_sample"],
            max_decoded_bytes=cfg["max_geff_decoded_bytes_per_sample"],
            request_interval_seconds=cfg["request_interval_seconds"],
            retry_delays_seconds=cfg["retry_delays_seconds"],
            max_cumulative_retry_sleep_seconds=cfg[
                "max_cumulative_retry_sleep_seconds"
            ],
        )
        stats = negative_link_opportunities(
            acquired["nodes"],
            acquired["edges"],
            pair_radius_um=cfg["pair_radius_um"],
            spacing_um=cfg["spacing_um"],
        )
        embryo = sample_id.split("_", 1)[0]
        rows.append({
            "sample_id": sample_id,
            "embryo_id": embryo,
            **stats,
            "estimated_number_of_nodes": find_recursive(
                acquired["root_meta"],
                "estimated_number_of_nodes",
            ),
            "geff_files": acquired["planned_files"],
            "downloaded_files": acquired["downloaded_files"],
            "reused_files": acquired["reused_files"],
            "decoded_geff_bytes": acquired["decoded_bytes"],
        })

        scout_root = root / "data/multisample_scout"
        total_cache_bytes = sum(
            p.stat().st_size
            for p in scout_root.rglob("*")
            if p.is_file()
        )
        if total_cache_bytes > cfg["max_total_scout_bytes"]:
            raise RuntimeError(
                f"Scout cache exceeded cap {cfg['max_total_scout_bytes']}."
            )

    table = pd.DataFrame(rows)
    table["meets_link_supervision_target"] = (
        (table["positive_link_opportunities"] >= cfg["min_positive_opportunities"])
        & (table["negative_link_opportunities"] >= cfg["min_negative_opportunities"])
    )
    table = table.sort_values(
        [
            "meets_link_supervision_target",
            "negative_link_opportunities",
            "division_sources",
            "positive_link_opportunities",
            "max_simultaneous_annotated_nodes",
            "nodes",
        ],
        ascending=False,
    ).reset_index(drop=True)
    atomic(out / "sample_ranking.csv", table.to_csv(index=False))

    shortlist = table.head(min(len(table), int(cfg["shortlist_size"]))).copy()
    shortlist["shortlist_reason"] = np.where(
        shortlist["meets_link_supervision_target"],
        "meets_positive_and_negative_opportunity_target",
        "best_available_cross_embryo",
    )
    atomic(out / "shortlist.csv", shortlist.to_csv(index=False))

    receipt = {
        "stage": "multisample_geff_scout",
        "status": "completed",
        "samples_scouted": int(len(table)),
        "distinct_embryos_scouted": int(table["embryo_id"].nunique()),
        "samples_meeting_link_supervision_target": int(
            table["meets_link_supervision_target"].sum()
        ),
        "shortlist_size": int(len(shortlist)),
        "max_negative_link_opportunities": int(
            table["negative_link_opportunities"].max()
        ),
        "max_positive_link_opportunities": int(
            table["positive_link_opportunities"].max()
        ),
        "max_division_sources": int(table["division_sources"].max()),
        "total_scout_cache_bytes": int(total_cache_bytes),
        "training_fits": 0,
        "image_downloads": 0,
        "official_score": None,
        "evaluation": cfg["evaluation_scope"],
        "crc32c_backend": backend,
        "next_decision": (
            "download_images_for_shortlisted_samples"
            if int(table["meets_link_supervision_target"].sum()) > 0
            else "switch_to_sparse_label_aware_association_objective"
        ),
    }
    write_json(out / "receipt.json", receipt)
    return receipt
