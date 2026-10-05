"""Employer-facing portfolio integrity checks."""
from pathlib import Path
import json
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "PUBLICATION_MANIFEST.json").read_text())

def require(ok, message):
    if not ok:
        raise ValueError(message)

for rel in MANIFEST["required_files"]:
    require((ROOT / rel).is_file(), f"Missing required portfolio file: {rel}")

for rel in MANIFEST["required_directories"]:
    require((ROOT / rel).is_dir(), f"Missing required portfolio directory: {rel}")

obsolete = [
    "PROJECT_SUMMARY_AND_RESTART_PLAN.md",
    "archive/final_sprint_source_snapshot.json",
    "scripts/extract_final_sprint_source.py",
    "docs/FINAL_SPRINT_REPRODUCTION.md",
    "docs/evidence/2026-09-29_final_project_handoff.md",
    "docs/evidence/2026-09-29_successor_final_submit.md",
    "docs/evidence/2026-09-29_successor_upload_continuation.md",
]
for rel in obsolete:
    require(not (ROOT / rel).exists(), f"Internal handoff artifact should not be public: {rel}")

primary_docs = [
    "README.md", "CASE_STUDY.md", "RESULTS.md", "REPRODUCE.md",
    "REPRODUCIBILITY.md", "FRONTIER.md", "docs/ENGINEERING_OVERVIEW.md",
    "docs/STATUS.md", "docs/PUBLICATION_SCOPE.md", "notebooks/README.md",
]

forbidden_framing = re.compile(
    r"\bleaderboard\b|\btop score\b|\brecorded gap\b|\bleader snapshot\b|"
    r"\bbeat the\b|\bfinal sprint\b|\brestart plan\b",
    re.IGNORECASE,
)
secret_like = re.compile(
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|"
    r"github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)

for rel in primary_docs:
    path = ROOT / rel
    text = path.read_text()
    require(not forbidden_framing.search(text), f"Ranking/internal framing leaked into {rel}")
    require(not secret_like.search(text), f"Credential-like material in {rel}")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        local = target.split("#")[0]
        if not local:
            continue
        require((path.parent / local).exists(), f"Broken link in {rel}: {target}")

readme = (ROOT / "README.md").read_text()
for phrase in [
    "60-second employer review",
    "System architecture",
    "Selected engineering and research outcomes",
    "Evidence discipline",
    "Repository map",
    "What is intentionally not published",
    "34.7×",
    "557",
    "45",
]:
    require(phrase in readme, f"README missing employer-facing element: {phrase}")

require("```mermaid" in readme, "README missing architecture diagram")
require("docs/ENGINEERING_OVERVIEW.md" in readme, "README missing engineering-review path")
require("0.953" not in readme, "README should not center unverified public-score lineage")

summary = json.loads((ROOT / "reports/portfolio_summary.json").read_text())
for key in ["leader_snapshot_score", "leader_snapshot_observed_utc", "recorded_gap", "independently_recreated_leader_training"]:
    require(key not in summary, f"Ranking-gap field remains in portfolio summary: {key}")
require(summary["official_public_score"] == 0.947, "Reference evidence contract changed")
require(summary["completed_scientific_workflows"] == 45, "Scientific workflow count changed")
require(summary["executed_evidence_notebooks"] == 7, "Notebook count changed")
require(summary["association_features_evaluated"] == 557, "Feature-study count changed")

hero = (ROOT / "assets/exact_error_budget_0.png").read_bytes()
require(hero[:8] == b"\x89PNG\r\n\x1a\n", "Hero evidence image is not a PNG")
width, height = struct.unpack(">II", hero[16:24])
require(width >= 600 and height >= 300, f"Hero evidence image too small: {width}x{height}")

print(json.dumps({
    "status": "passed",
    "release_id": MANIFEST["release_id"],
    "primary_docs_checked": len(primary_docs),
    "required_files_checked": len(MANIFEST["required_files"]),
    "ranking_gap_fields_present": 0,
    "broken_links": 0,
    "hero_image": [width, height],
}, indent=2))
