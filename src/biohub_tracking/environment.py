"""Read local capacity and selected package versions; never dump credentials or env vars."""
from __future__ import annotations
from importlib import metadata
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import psutil
from .io import utc_now

PACKAGES = ("biohub-tracking-research", "numpy", "pandas", "plotly", "psutil",
            "ipykernel", "nbformat", "pytest", "kaggle")


def snapshot(root: Path) -> dict:
    root = Path(root).resolve()
    disk = shutil.disk_usage(root)
    vm = psutil.virtual_memory()
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        git = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=5)
        head = git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        head = None
    checks = {
        "supported_python_3_11_to_3_13": (3, 11) <= sys.version_info[:2] < (3, 14),
        "project_venv_kernel": Path(sys.prefix).resolve() == (root / ".venv").resolve(),
        "required_packages_present": all(v is not None for v in versions.values()),
        "at_least_10_GiB_free_for_setup_only": disk.free >= 10 * 1024**3,
    }
    return {
        "utc": utc_now(), "purpose": "setup_only_not_training_readiness",
        "python": platform.python_version(), "platform": platform.system(),
        "cpu_logical": psutil.cpu_count(logical=True),
        "memory_total_GiB": round(vm.total / 1024**3, 3),
        "memory_available_GiB": round(vm.available / 1024**3, 3),
        "disk_total_GiB": round(disk.total / 1024**3, 3),
        "disk_free_GiB": round(disk.free / 1024**3, 3),
        "package_versions": versions, "git_head": head,
        "git_status_verified": False,
        "checks": checks, "setup_checks_passed": all(checks.values()),
        "credentials_recorded": False,
    }
