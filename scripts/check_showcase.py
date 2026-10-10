"""Dependency-free checks of public evidence, navigation and demo scope."""
from pathlib import Path
import json
import re
import struct
import xml.etree.ElementTree as ET

from verify_portfolio import SECRET, check_public_metadata, require

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_DOCS = [
    "README.md", "CASE_STUDY.md", "RESULTS.md", "REPRODUCE.md",
    "REPRODUCIBILITY.md", "FRONTIER.md", "docs/ENGINEERING_OVERVIEW.md",
    "docs/STATUS.md", "docs/PUBLICATION_SCOPE.md", "notebooks/README.md",
]
FORBIDDEN_FRAMING = re.compile(
    r"\bleaderboard\b|\btop score\b|\brecorded gap\b|\bleader snapshot\b|"
    r"\bbeat the\b|\bfinal sprint\b|\brestart plan\b", re.IGNORECASE,
)


def check_local_links(root, rel, text):
    root = Path(root).resolve()
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        local = target.split("#")[0]
        if not local:
            continue
        path = (root / rel).parent / local
        require(path.resolve().is_relative_to(root), f"Link escapes repository in {rel}: {target}")
        require(path.exists(), f"Broken link in {rel}: {target}")


def check_svg(path):
    tree = ET.fromstring(Path(path).read_text())
    ns = "{http://www.w3.org/2000/svg}"
    require(tree.tag == ns + "svg", "Invalid SVG root")
    require(tree.find(ns + "title") is not None and tree.find(ns + "desc") is not None,
            "SVG requires accessible title and description")
    require(tree.attrib.get("role") == "img", "SVG requires image role")
    for element in tree.iter():
        require(element.tag not in {ns + "script", ns + "foreignObject"}, "Active SVG content")
        for key, value in element.attrib.items():
            require(not key.lower().startswith("on"), "SVG event handler")
            if key.endswith("href"):
                require(value.startswith("#"), "External SVG dependency")


def check_showcase(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / "PUBLICATION_MANIFEST.json").read_text())
    for rel in manifest["required_files"]:
        require((root / rel).is_file(), "Missing required portfolio file: " + rel)
    for rel in manifest["required_directories"]:
        require((root / rel).is_dir(), "Missing required portfolio directory: " + rel)
    for rel in (
        "PROJECT_SUMMARY_AND_RESTART_PLAN.md", "archive/final_sprint_source_snapshot.json",
        "scripts/extract_final_sprint_source.py", "docs/FINAL_SPRINT_REPRODUCTION.md",
        "docs/evidence/2026-09-29_final_project_handoff.md",
        "docs/evidence/2026-09-29_successor_final_submit.md",
        "docs/evidence/2026-09-29_successor_upload_continuation.md",
    ):
        require(not (root / rel).exists(), "Internal handoff artifact should not be public: " + rel)
    for rel in PRIMARY_DOCS:
        text = (root / rel).read_text()
        require(not FORBIDDEN_FRAMING.search(text), "Ranking/internal framing leaked into " + rel)
        require(not SECRET.search(text), "Credential-like material in " + rel)
        check_public_metadata(text, rel)
        check_local_links(root, rel, text)

    # Evidence and entry points are durable contracts; exact section headings are not.
    readme = (root / "README.md").read_text()
    for target in ("CASE_STUDY.md", "RESULTS.md", "REPRODUCE.md", "docs/ENGINEERING_OVERVIEW.md", "public-demo/index.html"):
        require(target in readme or (target == "public-demo/index.html" and "public-demo" in readme),
                "README missing review route: " + target)
    for evidence in ("0.947", "0.946", "557", "45", "34.7"):
        require(evidence in readme, "README missing measured evidence: " + evidence)
    require("synthetic" in readme.lower() and "private" in readme.lower(), "README must explain public reproduction scope")
    require("```mermaid" in readme or "architecture.svg" in readme, "README missing architecture diagram")
    for rel in ("docs/assets/hero.svg", "docs/assets/architecture.svg"):
        check_svg(root / rel)

    summary = json.loads((root / "reports/portfolio_summary.json").read_text())
    check_public_metadata(summary)
    require(summary["official_public_score"] == 0.947, "Reference evidence contract changed")
    require(summary["completed_scientific_workflows"] == 45, "Scientific workflow count changed")
    require(summary["executed_evidence_notebooks"] == 7, "Notebook count changed")
    require(summary["association_features_evaluated"] == 557, "Feature-study count changed")

    data = json.loads((root / "public-demo/data.json").read_text())
    provenance = data.get("provenance", {})
    require(provenance.get("evidence_type") == "SYNTHETIC_ONLY", "Public demo must identify synthetic evidence")
    require("not official" in provenance.get("metric_scope", "").lower(), "Public demo must distinguish its diagnostics")
    require(provenance.get("private_assets_used") is False and provenance.get("official_score") is None,
            "Synthetic demo must not contain private assets or an official score")
    for rel in ("public-demo/index.html", "public-demo/app.js", "public-demo/styles.css", "public-demo/data.js", "public-demo/graph.js"):
        require((root / rel).is_file(), "Missing public demo file: " + rel)

    image = (root / "assets/exact_error_budget_0.png").read_bytes()
    require(image[:8] == b"\x89PNG\r\n\x1a\n", "Historical evidence image is not PNG")
    width, height = struct.unpack(">II", image[16:24])
    require(width >= 600 and height >= 300, "Historical evidence image too small")
    return {"status": "passed", "release_id": manifest["release_id"],
            "primary_docs_checked": len(PRIMARY_DOCS), "required_files_checked": len(manifest["required_files"]),
            "ranking_gap_fields_present": 0, "broken_links": 0, "hero_image": [width, height],
            "accessible_svg_assets": 2, "demo_evidence": provenance["evidence_type"]}


if __name__ == "__main__":
    print(json.dumps(check_showcase(), indent=2))
