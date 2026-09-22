from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"crc32c usage is deprecated since numcodecs v0\.16\.4\..*",
        category=DeprecationWarning,
    )
    import zarr

ARRAYS = {
    "node_ids": "nodes/ids",
    "t": "nodes/props/t/values",
    "z": "nodes/props/z/values",
    "y": "nodes/props/y/values",
    "x": "nodes/props/x/values",
    "edges": "edges/ids",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _recursive_find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _recursive_find(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _recursive_find(value, key)
            if found is not None:
                return found
    return None


def _dtype_itemsize(dtype_value) -> int:
    try:
        return np.dtype(dtype_value).itemsize
    except Exception as exc:
        raise ValueError(f"Unsupported Zarr data_type: {dtype_value!r}") from exc


def array_plan(metadata: dict) -> dict:
    if metadata.get("node_type") != "array":
        raise ValueError("Expected Zarr v3 array metadata.")
    shape = tuple(int(x) for x in metadata["shape"])
    chunk_shape = tuple(int(x) for x in metadata["chunk_grid"]["configuration"]["chunk_shape"])
    if len(shape) != len(chunk_shape):
        raise ValueError("shape/chunk_shape dimensionality mismatch")
    counts = tuple(math.ceil(s / c) for s, c in zip(shape, chunk_shape, strict=True))

    enc = metadata.get("chunk_key_encoding", {})
    name = enc.get("name", "default")
    sep = enc.get("configuration", {}).get("separator", "/")
    coords = list(itertools.product(*[range(n) for n in counts]))
    keys = []
    for coord in coords:
        if name == "default":
            keys.append("c" + sep + sep.join(str(v) for v in coord))
        elif name == "v2":
            keys.append(sep.join(str(v) for v in coord))
        else:
            raise ValueError(f"Unsupported chunk key encoding: {name!r}")
    decoded_bytes = int(np.prod(shape)) * _dtype_itemsize(metadata["data_type"])
    return {
        "shape": shape,
        "chunk_shape": chunk_shape,
        "chunk_counts": counts,
        "chunk_keys": keys,
        "decoded_bytes": decoded_bytes,
    }


def resolve_kaggle_command(
    *,
    prefix_value=None,
    python_executable=None,
    path_value=None,
    project_root=None,
):
    # Do not resolve virtualenv Python symlinks before checking their environment.
    import os
    import shutil
    import sys
    from pathlib import Path

    prefix = Path(prefix_value or sys.prefix)
    py = Path(python_executable or sys.executable)
    project = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).absolute().parents[2]
    )

    executable_candidates = [
        project / ".venv" / "bin" / "kaggle",
        prefix / "bin" / "kaggle",
        py.with_name("kaggle"),
        Path.home() / ".local" / "bin" / "kaggle",
        Path("/opt/conda/bin/kaggle"),
    ]
    for candidate in executable_candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return [str(candidate)]

    found = shutil.which("kaggle", path=path_value)
    if found:
        return [found]

    python_candidates = [
        project / ".venv" / "bin" / "python",
        prefix / "bin" / "python",
        py,
    ]
    for candidate in python_candidates:
        if candidate.is_file():
            return [str(candidate), "-m", "kaggle"]

    raise FileNotFoundError(
        "No Kaggle command could be constructed. "
        f"project={project} sys.prefix={prefix} python={py}"
    )


def _download_one(
    *,
    competition: str,
    remote: str,
    destination: Path,
    timeout_seconds: int = 45,
) -> Path:
    """Download one Kaggle competition file into its exact nested local path.

    Kaggle CLI 2.2.4 flattens nested ``-f`` paths to the downloaded basename.
    Pointing ``-p`` at the target's parent directory makes that flattened
    basename land at the exact local path we want.
    """
    import shutil
    import zipfile

    destination = Path(destination)
    target = destination / remote
    if target.is_file() and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)

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
    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-1600:]
        raise RuntimeError(
            f"Kaggle download failed for {remote!r}: {tail}"
        )

    if target.is_file() and target.stat().st_size > 0:
        return target

    # Some Kaggle single-file responses are server-wrapped as <basename>.zip.
    zip_candidate = target.with_name(target.name + ".zip")
    if zip_candidate.is_file() and zip_candidate.stat().st_size > 0:
        with zipfile.ZipFile(zip_candidate) as archive:
            safe_members = [
                m for m in archive.infolist()
                if not m.is_dir()
                and Path(m.filename).name == target.name
                and ".." not in Path(m.filename).parts
            ]
            if len(safe_members) != 1:
                raise FileNotFoundError(
                    "Kaggle returned a ZIP wrapper, but the requested file "
                    f"could not be identified uniquely: remote={remote!r}, "
                    f"members={[m.filename for m in archive.infolist()[:20]]!r}"
                )
            member = safe_members[0]
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        zip_candidate.unlink()
        if target.is_file() and target.stat().st_size > 0:
            return target

    # Collect bounded diagnostics from the directory where Kaggle was asked to write.
    observed = []
    for p in sorted(target.parent.iterdir()):
        if p.is_file():
            observed.append(
                {
                    "name": p.name,
                    "bytes": p.stat().st_size,
                }
            )
        if len(observed) >= 20:
            break

    raise FileNotFoundError(
        "Kaggle command returned success but the requested file did not land "
        "at the expected nested target after basename-aware download handling. "
        f"remote={remote!r} expected={target} "
        f"download_dir={target.parent} observed={observed!r}"
    )



def acquire_sample(
    *,
    root: Path,
    competition: str,
    sample_id: str,
    decoded_cap_bytes: int = 64 * 1024**2,
    max_files: int = 200,
) -> dict:
    base_remote = f"train/{sample_id}.geff"
    cache_root = root / "data" / "labeled_snapshot"
    cache_root.mkdir(parents=True, exist_ok=True)

    required_metadata = [f"{base_remote}/zarr.json"]
    required_metadata += [f"{base_remote}/{rel}/zarr.json" for rel in ARRAYS.values()]

    downloaded = []
    for remote in required_metadata:
        path = _download_one(
            competition=competition,
            remote=remote,
            destination=cache_root,
        )
        downloaded.append(path)

    plans = {}
    decoded_total = 0
    chunk_remotes = []
    local_geff = cache_root / base_remote
    for name, rel in ARRAYS.items():
        meta_path = local_geff / rel / "zarr.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        plan = array_plan(meta)
        plans[name] = plan
        decoded_total += plan["decoded_bytes"]
        chunk_remotes.extend(f"{base_remote}/{rel}/{key}" for key in plan["chunk_keys"])

    if decoded_total > decoded_cap_bytes:
        raise RuntimeError(
            f"STOP: GEFF decoded arrays would use {decoded_total / 1024**2:.2f} MiB, "
            f"above cap {decoded_cap_bytes / 1024**2:.2f} MiB."
        )
    total_files = len(required_metadata) + len(chunk_remotes)
    if total_files > max_files:
        raise RuntimeError(f"STOP: planned {total_files} GEFF files, above cap {max_files}.")

    for remote in chunk_remotes:
        path = _download_one(
            competition=competition,
            remote=remote,
            destination=cache_root,
        )
        downloaded.append(path)

    arrays = {}
    for name, rel in ARRAYS.items():
        arrays[name] = np.asarray(zarr.open_array(local_geff / rel, mode="r")[:])

    n = len(arrays["node_ids"])
    if not all(len(arrays[k]) == n for k in ("t", "z", "y", "x")):
        raise ValueError("GEFF node property lengths do not match node_ids.")
    if arrays["edges"].ndim != 2 or arrays["edges"].shape[1] != 2:
        raise ValueError(f"Expected edges shape (N,2), got {arrays['edges'].shape}.")

    nodes = pd.DataFrame({
        "gt_id": arrays["node_ids"].astype(np.int64),
        "t": arrays["t"].astype(np.int64),
        "z": arrays["z"].astype(float),
        "y": arrays["y"].astype(float),
        "x": arrays["x"].astype(float),
    })
    edges = pd.DataFrame({
        "source_gt_id": arrays["edges"][:, 0].astype(np.int64),
        "target_gt_id": arrays["edges"][:, 1].astype(np.int64),
    })

    root_meta = json.loads((local_geff / "zarr.json").read_text(encoding="utf-8"))
    estimated_nodes = _recursive_find(root_meta, "estimated_number_of_nodes")

    out = root / "outputs" / "labeled"
    out.mkdir(parents=True, exist_ok=True)
    nodes_path = out / "gt_nodes.csv"
    edges_path = out / "gt_edges.csv"
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)

    manifest = {
        "sample_id": sample_id,
        "competition": competition,
        "status": "completed",
        "network_scope": "sparse GEFF labels only; no image download",
        "downloaded_file_count": len(downloaded),
        "decoded_geff_bytes": decoded_total,
        "gt_nodes": len(nodes),
        "gt_edges": len(edges),
        "estimated_number_of_nodes": estimated_nodes,
        "arrays": {
            k: {
                "shape": list(v["shape"]),
                "chunk_shape": list(v["chunk_shape"]),
                "chunk_count": len(v["chunk_keys"]),
                "decoded_bytes": v["decoded_bytes"],
            }
            for k, v in plans.items()
        },
        "artifacts": {
            str(nodes_path.relative_to(root)): sha256_file(nodes_path),
            str(edges_path.relative_to(root)): sha256_file(edges_path),
        },
    }
    manifest_path = out / "label_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
