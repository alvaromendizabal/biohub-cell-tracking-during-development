from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _find_notebook(directory: Path) -> Path:
    paths = sorted(Path(directory).glob("*.ipynb"))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one notebook under {directory}, found {paths}"
        )
    return paths[0]


def extract_notebook_source(notebook_path: Path) -> dict:
    nb = read_json(notebook_path)
    code_cells = [
        c for c in nb.get("cells", [])
        if c.get("cell_type") == "code"
    ]
    markdown_cells = [
        c for c in nb.get("cells", [])
        if c.get("cell_type") == "markdown"
    ]

    def src(cell):
        value = cell.get("source", "")
        if isinstance(value, list):
            return "".join(value)
        return str(value)

    code = "\n\n".join(src(c) for c in code_cells)
    markdown = "\n\n".join(src(c) for c in markdown_cells)

    return {
        "notebook": nb,
        "code": code,
        "markdown": markdown,
        "code_cells": len(code_cells),
        "markdown_cells": len(markdown_cells),
        "code_lines": len(code.splitlines()),
    }


def audit_public_sources(root: Path) -> dict:
    root = Path(root).resolve()
    cfg = read_json(
        root / "configs/focus3d_dense_supervision_source_audit.json"
    )
    refs = root / "references/public_notebooks/focus3d_dense_supervision"
    out = root / "outputs/focus3d_source_audit"
    out.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    component_rows = []
    extracted_root = out / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)

    for source_cfg in cfg["public_sources"]:
        key = str(source_cfg["key"])
        src_dir = refs / key
        notebook = _find_notebook(src_dir)
        extracted = extract_notebook_source(notebook)

        code = extracted["code"]
        markdown = extracted["markdown"]
        searchable = (code + "\n" + markdown).lower()

        py_path = extracted_root / f"{key}.py"
        py_path.write_text(code, encoding="utf-8")

        manifest_rows.append(
            {
                "key": key,
                "kaggle_kernel": source_cfg["kaggle_kernel"],
                "notebook_path": str(notebook.relative_to(root)),
                "sha256": sha256(notebook),
                "bytes": int(notebook.stat().st_size),
                "code_cells": int(extracted["code_cells"]),
                "markdown_cells": int(extracted["markdown_cells"]),
                "code_lines": int(extracted["code_lines"]),
                "license_evidence": source_cfg["license_evidence"],
            }
        )

        for family, needles in cfg["source_audit_signals"].items():
            hits = []
            for needle in needles:
                count = searchable.count(str(needle).lower())
                if count:
                    hits.append(f"{needle}:{count}")
            component_rows.append(
                {
                    "key": key,
                    "family": family,
                    "present": bool(hits),
                    "hit_count": int(
                        sum(int(h.split(":")[-1]) for h in hits)
                    ),
                    "hits": "; ".join(hits),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    components = pd.DataFrame(component_rows)

    # Transferability gate: the distilled detector source must expose
    # point-detector + dense-supervision signals; direct FOCUS3D source must
    # expose FOCUS3D + physical-coordinate signals.
    def present(key, family):
        row = components[
            (components["key"] == key)
            & (components["family"] == family)
        ]
        return bool(len(row) and bool(row.iloc[0]["present"]))

    required = {
        "cell_point_detector.point_detector": present(
            "cell_point_detector", "point_detector"
        ),
        "cell_point_detector.dense_supervision": present(
            "cell_point_detector", "dense_supervision"
        ),
        "cell_point_detector.physical_coordinates": present(
            "cell_point_detector", "physical_coordinates"
        ),
        "focus3d_direct_submit.focus3d": present(
            "focus3d_direct_submit", "focus3d"
        ),
        "focus3d_direct_submit.physical_coordinates": present(
            "focus3d_direct_submit", "physical_coordinates"
        ),
    }

    transferable = all(required.values())

    manifest.to_csv(out / "source_manifest.csv", index=False)
    components.to_csv(out / "component_matrix.csv", index=False)

    receipt = {
        "status": "completed",
        "sources": int(len(manifest)),
        "required_signals": required,
        "transferable_source_path": transferable,
        "model_downloads": 0,
        "gpu_jobs": 0,
        "training_fits": 0,
        "feature_recompute": False,
        "next_decision": (
            cfg["next_gate"]["if_transferable"]
            if transferable
            else cfg["next_gate"]["if_not_transferable"]
        ),
    }
    (out / "audit_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt
