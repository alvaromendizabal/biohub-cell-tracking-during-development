#!/usr/bin/env python3
"""Generate the authored synthetic movie and real constrained-solver demo offline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_tracking.synthetic_demo import DemoConfig, write_demo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "public-demo")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        result = write_demo(args.output, DemoConfig(seed=args.seed))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Synthetic demo stopped: {exc}\n")
    print(json.dumps({"evidence_type": result["provenance"]["evidence_type"],
                      "summary": result["summary"], "default_solution": result["default_solution"],
                      "data_json": str(args.output / "data.json"), "data_js": str(args.output / "data.js")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
