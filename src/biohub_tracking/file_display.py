"""Small file-backed notebook outputs; no feature computation or browser polling.

The heavy Plotly page stays in an HTML file. JupyterLab resolves the relative
iframe URL against the notebook location. The notebook and outputs/ directory
must travel together. No server configuration, ports, authentication, or CDN.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from datetime import datetime, timezone

VERSION = "biohub-file-display-20260914-1"
MAX_OUTPUT_BYTES = 2048
MAX_PAGE_BYTES = 12 * 1024**2


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Refusing a symlinked output destination.")
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: dict) -> None:
    atomic(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def display_signature(root: Path) -> str:
    root = Path(root)
    names = ["src/biohub_tracking/file_display.py", "tests/test_inline_display.py",
             "scripts/verify_inline_display.py"]
    hashes = {name: sha(root / name) for name in names}
    # Source-only hashes: saving actual notebook output does not stale the check.
    for name in ["05_plotly_display_check.ipynb", "06_morphology_feature_round.ipynb",
                 "07_association_feature_round.ipynb"]:
        notebook = json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
        sources = [(c["cell_type"], "".join(c.get("source", []))) for c in notebook["cells"]]
        # The user's explicit visual confirmation values are not executable pipeline code.
        if name.startswith("05_"):
            sources = [item for item in sources if "SAVED_REOPENED_AND_HOVERED =" not in item[1]]
        hashes["notebooks/" + name] = hashlib.sha256(json.dumps(sources).encode()).hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def check_display_tests(root: Path) -> None:
    path = Path(root) / "outputs/features/visualization/inline_tests_receipt.json"
    if not path.is_file():
        raise ValueError("Run scripts/verify_inline_display.py once before opening the viewer.")
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "passed" or receipt.get("signature") != display_signature(root):
        raise ValueError("The display source check is missing or stale. Run scripts/verify_inline_display.py.")


def iframe_markup(relative_url: str, *, label: str = "Interactive Plotly charts", height: int = 720) -> str:
    """Accept only relative project-output paths, never inline documents or URLs."""
    if not isinstance(relative_url, str) or len(relative_url) > 512:
        raise ValueError("Unexpected iframe path.")
    if any(c in relative_url for c in ("\\", ":", "?", "#", "\x00", "\r", "\n")):
        raise ValueError("The iframe must reference a local HTML artifact.")
    parts = PurePosixPath(relative_url).parts
    if parts[:3] != ("..", "outputs", "features") or ".." in parts[1:] or not relative_url.endswith(".html"):
        raise ValueError("The iframe must stay under ../outputs/features/.")
    if not isinstance(height, int) or not 400 <= height <= 1400 or len(label) > 160:
        raise ValueError("Invalid display dimensions or label.")
    url = html.escape(quote(relative_url, safe="/"), quote=True)
    title = html.escape(label, quote=True)
    markup = (
        '<div data-biohub-inline="file-backed"><p><strong>' + title + '</strong></p>'
        '<iframe src="' + url + '" title="' + title + '" width="100%" height="' + str(height) + '" '
        'sandbox="allow-scripts allow-downloads" referrerpolicy="same-origin" '
        'style="border:1px solid;display:block"></iframe>'
        '<p>If this remains blank for 30 seconds, stop; do not rerun features. '
        'The notebook and its outputs folder must remain together.</p></div>'
    )
    if len(markup.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise ValueError("Display output exceeds its two-KiB limit.")
    return markup


def _publish(root: Path, page: str, name: str, *, count: int, signature: str, label: str) -> None:
    """Write the page, send a tiny output, and record return separately from visibility."""
    from IPython.display import display
    root = Path(root).resolve()
    if len(page.encode("utf-8")) > MAX_PAGE_BYTES:
        raise ValueError("HTML artifact is larger than the bounded viewer supports.")
    if name not in ("pilot_one", "pilot_gallery", "round1_gallery", "round2_gallery"):
        raise ValueError("Unsupported gallery name.")
    relative = f"outputs/features/visualization/file_backed/{name}.html"
    path = root / relative
    atomic(path, page)
    markup = iframe_markup("../" + relative, label=label, height=740 if count == 1 else 1100)
    receipt_path = path.with_suffix(".json")
    receipt = {"utc": utc(), "version": VERSION, "phase": "prepared",
               "html": relative, "html_sha256": sha(path), "html_bytes": path.stat().st_size,
               "output_bytes": len(markup.encode()), "figure_count": count,
               "signature": signature, "display_source_signature": display_signature(root),
               "display_request_returned": False, "visual_success": None,
               "network_requests_from_python": 0, "feature_computation": False,
               "note": "Browser fetches the local HTML through the existing Jupyter session. Visibility requires user confirmation."}
    write_json(receipt_path, receipt)
    print(f"{name.upper()}_HTML_READY output_bytes={len(markup.encode())} figures={count}", flush=True)
    display({"text/html": markup, "text/plain": label + " (local HTML iframe)"}, raw=True)
    receipt.update(phase="display_call_returned", display_request_returned=True, utc=utc())
    write_json(receipt_path, receipt)
    print(f"{name.upper()}_CELL_RETURNED — Python returned; browser visibility is separate.", flush=True)


def show_pilot_one(root: Path) -> None:
    check_display_tests(root)
    from .feature_display import visual_sources, build_page
    specs, js, signature = visual_sources(root)
    # Existing, hash-verified first saved figure: 03_voxel_spacing.plotly.json.
    page = build_page(specs[:1], js, "Biohub | single saved chart", "Existing voxel-spacing evidence; no new computation.")
    _publish(root, page, "pilot_one", count=1, signature=signature, label="One saved Plotly chart — hover before continuing")


def show_pilot_gallery(root: Path) -> None:
    check_display_tests(root)
    from .feature_display import visual_sources, build_page
    specs, js, signature = visual_sources(root)
    if len(specs) != 8:
        raise ValueError("Expected the eight already-saved pilot figures.")
    page = build_page(specs, js, "Biohub | completed pilot",
                      "39 candidates; 74 candidate links. No model fits or official score. Original 3-D chart uses an XY compatibility projection.",
                      signature[:8].upper())
    _publish(root, page, "pilot_gallery", count=len(specs), signature=signature, label="Eight saved pilot charts — scroll inside the panel")


def attest_visible(root: Path, code: str, saved_reopened_and_hovered: bool = False) -> None:
    check_display_tests(root)
    root = Path(root)
    from .feature_display import visual_sources
    _, _, signature = visual_sources(root)
    request_path = root / "outputs/features/visualization/file_backed/pilot_gallery.json"
    request = json.loads(request_path.read_text())
    if (not saved_reopened_and_hovered or str(code).strip().upper() != signature[:8].upper()
        or request.get("signature") != signature or request.get("figure_count") != 8
        or request.get("display_request_returned") is not True
        or request.get("display_source_signature") != display_signature(root)
        or sha(root / request["html"]) != request["html_sha256"]):
        raise ValueError("Confirm the actual eight-chart display after saving/reopening. Do not bypass a blank output.")
    # Compatible with the existing workers; its extra fields document the new display path.
    write_json(root / "outputs/features/visualization/visual_gate.json", {
        "utc": utc(), "signature": signature, "saved_reopened_and_hovered": True,
        "figure_count": 8, "verification_type": "user_attestation_file_backed_inline",
        "display_source_signature": display_signature(root), "version": VERSION,
        "html": request["html"], "html_sha256": request["html_sha256"]})
    print("PLOTLY_VISUAL_GATE_RECORDED — user confirmation, not an automated browser test.", flush=True)


def require_visible(root: Path) -> dict:
    check_display_tests(root)
    from .feature_display import require_visible as original_gate
    gate = original_gate(root)
    if gate.get("display_source_signature") != display_signature(root):
        raise ValueError("Complete the revised notebook 05 display gate first.")
    if sha(Path(root) / gate["html"]) != gate["html_sha256"]:
        raise ValueError("The visually confirmed HTML artifact changed; reopen notebook 05.")
    return gate


def show_round_gallery(root: Path, stage: str) -> None:
    """Reuse completed, compatible tables; do not call a feature worker."""
    require_visible(root)
    if stage not in ("round1", "round2"):
        raise ValueError("Only round1 and round2 are supported.")
    from .feature_display import round_figures, visual_sources, build_page
    root = Path(root)
    specs, receipt = round_figures(root, stage)
    _, js, _ = visual_sources(root)
    directory = root / "outputs/features/figures" / stage
    # Individual exports share a local JS file; the gallery is self-contained on disk.
    atomic(directory / "plotly.min.js", js)
    for i, spec in enumerate(specs, 1):
        name = f"{stage}_{i:02d}"
        atomic(directory / (name + ".plotly.json"), json.dumps(spec) + "\n")
        atomic(directory / (name + ".html"), build_page([spec], None, name, "Actual pilot features; no measured predictive score.", external_js=True))
    page = build_page(specs, js, "Biohub | " + stage, f'8 families; 128 descriptors; {receipt["rows"]} rows; no model fits; official score unmeasured.')
    atomic(directory / "gallery.html", page)
    _publish(root, page, stage + "_gallery", count=len(specs), signature=receipt["signature"], label=stage + " — eight inline Plotly charts")
