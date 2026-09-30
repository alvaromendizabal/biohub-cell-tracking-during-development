"""Parse one Kaggle CLI metadata page. NEVER treat a page as a complete inventory."""
from __future__ import annotations
import csv
import io
import math
import re
from pathlib import PurePosixPath
from .io import safe_relative_path


def size_bytes_estimate(value: str) -> int:
    """Convert CLI-rendered units; rounded human-readable sizes remain estimates."""
    text = str(value).strip().replace(",", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KIB|MIB|GIB|TIB)?", text, re.I)
    if not match:
        raise ValueError(f"Unsupported size representation: {text[:40]!r}")
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    power = {"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4,
             "KIB": 1, "MIB": 2, "GIB": 3, "TIB": 4}[unit]
    base = 1024 if "I" in unit else 1000
    result = number * base**power
    if not math.isfinite(result) or result < 0:
        raise ValueError("Nonfinite or negative size")
    return int(round(result))


def describe_path(value: str) -> dict:
    path = safe_relative_path(value)
    parts = PurePosixPath(path).parts
    split = next((p for p in parts if p in ("train", "test")), "other")
    stores = [p for p in parts if p.endswith((".zarr", ".geff"))]
    sample = stores[0].rsplit(".", 1)[0] if stores else ""
    kind = ("image_zarr" if any(p.endswith(".zarr") for p in stores) else
            "annotation_geff" if any(p.endswith(".geff") for p in stores) else
            "submission_example" if parts[-1] == "sample_submission.csv" else "other")
    return {"path": path, "split": split, "kind": kind,
            "sample_id": sample, "embryo_id": sample.split("_")[0] if sample else ""}


def parse_cli_csv(text: str, max_rows: int = 200) -> list[dict]:
    if len(text.encode("utf-8")) > 2 * 1024**2:
        raise ValueError("Metadata output exceeded the 2 MiB safety bound")
    lines = text.lstrip("\ufeff").splitlines()
    header_index = None
    columns = None
    for i, line in enumerate(lines):
        fields = [x.strip() for x in next(csv.reader([line]), [])]
        lowered = [x.lower() for x in fields]
        if "name" in lowered and "size" in lowered:
            header_index, columns = i, lowered
            break
    if header_index is None or columns is None:
        raise ValueError("Expected Kaggle CSV header with name,size; inspect the local raw page.")
    name_i, size_i = columns.index("name"), columns.index("size")
    records, seen = [], set()
    for fields in csv.reader(io.StringIO("\n".join(lines[header_index + 1:]))):
        if not fields or not any(x.strip() for x in fields):
            continue
        # CLI pagination notes are not file records.
        if len(fields) != len(columns):
            if len(fields) == 1 and ("page" in fields[0].lower() or "token" in fields[0].lower()):
                continue
            raise ValueError("Unexpected non-CSV content after the inventory header")
        value = fields[name_i].strip()
        if value in seen:
            raise ValueError("Duplicate path on one metadata page")
        record = describe_path(value)
        record["size_bytes_estimate"] = size_bytes_estimate(fields[size_i])
        records.append(record)
        seen.add(value)
        if len(records) > max_rows:
            raise ValueError("More file records returned than requested")
    if not records:
        raise ValueError("No file records returned; do not interpret this as an empty competition.")
    return records
