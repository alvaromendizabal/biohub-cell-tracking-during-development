"""Presentation-only integrity checks; no ML code or data execution."""
from pathlib import Path
import hashlib
import json
import re
import struct

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "PUBLICATION_MANIFEST.json").read_text())

assert manifest.get("hash_type") == "git_blob_sha1", manifest.get("hash_type")
expected = set(manifest["files"]) | {"PUBLICATION_MANIFEST.json"}
actual = {
    p.relative_to(root).as_posix()
    for p in root.rglob("*")
    if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
}
assert actual == expected, (actual - expected, expected - actual)

forbidden_parts = {
    "src", "research", "data", "models", "weights", ".venv",
    "outputs", "checkpoints", "artifacts", "cache", "caches",
}
forbidden_suffixes = {
    ".pt", ".pth", ".ckpt", ".pkl", ".joblib", ".npy", ".npz",
    ".pyz", ".zip", ".tar", ".gz",
}
sensitive_text = (
    "/home/sagemaker-user/",
    "aws_secret_access_key",
    "aws_session_token",
    "kaggle.json",
    "AKIA",
)

def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()

for name, expected_sha in manifest["files"].items():
    path = root / name
    data = path.read_bytes()
    assert not path.is_symlink(), name
    assert git_blob_sha1(data) == expected_sha, name
    assert not any(part in forbidden_parts for part in Path(name).parts), name
    assert path.suffix.lower() not in forbidden_suffixes, name

    if path.suffix == ".py":
        assert name == "scripts/check_showcase.py", name

    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text())
        assert notebook.get("metadata", {}).get("presentation_only") is True, name
        assert all(cell["cell_type"] == "markdown" for cell in notebook["cells"]), name

    if path.suffix == ".png":
        assert data[:8] == b"\x89PNG\r\n\x1a\n", name
        width, height = struct.unpack(">II", data[16:24])
        assert width >= 600 and height >= 300, (name, width, height)

    if path.suffix in {".md", ".py", ".yml", ".yaml", ".json", ".ipynb", ".txt"}:
        text = data.decode("utf-8")
        for marker in sensitive_text:
            assert marker not in text, (name, marker)

    if path.suffix == ".md":
        text = path.read_text()
        assert "```bash" not in text and "```python" not in text, name
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            assert (path.parent / target.split("#")[0]).exists(), (name, target)

readme = (root / "README.md").read_text()
frontier = (root / "FRONTIER.md").read_text()
results = (root / "RESULTS.md").read_text()
assert "0.947" in readme and "0.947" in results
assert "image-conditioned" in readme.lower() and "image-conditioned" in frontier.lower()
assert "Official improvements beyond 0.947" in results
print("PUBLIC_SHOWCASE_INTEGRITY_PASSED")
