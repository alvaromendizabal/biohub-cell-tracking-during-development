#!/usr/bin/env python3
"""Extract and validate the archived September 29 final-sprint source snapshot."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "archive" / "final_sprint_source_snapshot.json"
EXPECTED_SHA256 = "c1d96e516f778f1816784c7850e6c98fc3a046c5835553226f3bbdb1298622eb"
EXPECTED_FILES = 99


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_bundle():
    if not BUNDLE.is_file():
        raise FileNotFoundError(BUNDLE)
    if sha256(BUNDLE) != EXPECTED_SHA256:
        raise ValueError("final-sprint source snapshot hash mismatch")
    data = json.loads(BUNDLE.read_text())
    files = data.get("files", {})
    if len(files) != EXPECTED_FILES:
        raise ValueError(f"expected {EXPECTED_FILES} files, found {len(files)}")
    for rel, text in files.items():
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe archived path: {rel}")
        if not isinstance(text, str):
            raise TypeError(f"non-text archived member: {rel}")
        if p.suffix == ".py":
            ast.parse(text, filename=rel)
        elif p.suffix == ".json":
            json.loads(text)
        elif p.suffix == ".ipynb":
            nb = json.loads(text)
            if nb.get("nbformat") != 4:
                raise ValueError(f"unsupported notebook format: {rel}")
    return data


def extract(dest: Path):
    data = load_bundle()
    dest.mkdir(parents=True, exist_ok=True)
    for rel, text in data["files"].items():
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return len(data["files"])


def self_test():
    data = load_bundle()
    with tempfile.TemporaryDirectory() as td:
        n = extract(Path(td))
        if n != EXPECTED_FILES:
            raise AssertionError(n)
        py_count = sum(1 for p in Path(td).rglob("*.py"))
        if py_count < 20:
            raise AssertionError(f"unexpected Python source count: {py_count}")
    print(f"FINAL_SPRINT_SOURCE_SNAPSHOT_OK files={len(data['files'])}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    p = sub.add_parser("extract")
    p.add_argument("destination", type=Path)
    args = ap.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        n = extract(args.destination)
        print(f"EXTRACTED {n} files to {args.destination}")


if __name__ == "__main__":
    main()
