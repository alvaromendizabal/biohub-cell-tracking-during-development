"""Fresh-clone integrity checks for the reproducible Biohub archive."""
from __future__ import annotations
import ast, base64, binascii, json, re, struct, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".venv", ".venv-review"}
PRIVATE_PARTS = {".aws", ".kaggle", "data", "outputs", "artifacts", "checkpoints", "weights", "cache", "caches", "external", "repro_data"}
BAD_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".pkl", ".pickle", ".npz", ".npy", ".parquet", ".pyz", ".zip", ".gz", ".tar"}
SECRET = re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")

def require(ok, message):
    if not ok:
        raise ValueError(message)

def png_check(data: bytes):
    require(data[:8] == b"\x89PNG\r\n\x1a\n", "PNG signature")
    pos = 8
    dims = None
    found_end = False
    idat = []
    while pos < len(data):
        require(pos + 12 <= len(data), "PNG truncated")
        n = struct.unpack(">I", data[pos:pos+4])[0]
        kind = data[pos+4:pos+8]
        require(n <= 20_000_000 and pos + 12 + n <= len(data), "PNG chunk bound")
        body = data[pos+8:pos+8+n]
        crc = struct.unpack(">I", data[pos+8+n:pos+12+n])[0]
        require(binascii.crc32(kind + body) & 0xffffffff == crc, "PNG CRC")
        if kind == b"IHDR":
            dims = struct.unpack(">II", body[:8])
        if kind == b"IDAT":
            idat.append(body)
        pos += 12 + n
        if kind == b"IEND":
            found_end = True
            break
    require(dims and idat and found_end and pos == len(data), "PNG structure")
    raw = zlib.decompressobj().decompress(b"".join(idat), 50_000_001)
    require(0 < len(raw) <= 50_000_000, "PNG decoded bounds")
    return list(dims)

def check_path(rel: str):
    p = Path(rel)
    require(not p.is_absolute() and ".." not in p.parts, "Path escape " + rel)
    require(not (set(p.parts) & PRIVATE_PARTS), "Private runtime path " + rel)
    require(p.suffix.lower() not in BAD_EXT, "Private/binary artifact " + rel)
    require(not any(part.endswith((".zarr", ".geff")) or part.startswith(".venv") for part in p.parts), "Private store " + rel)
    require(p.name != "kaggle.json", "Credentials file")
    return True


def check_notebook(path: Path):
    nb = json.loads(path.read_text())
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    require(cells, "Notebook has no code " + str(path))
    counts = [c.get("execution_count") for c in cells]
    require(all(isinstance(x, int) and x > 0 for x in counts), "Unexecuted code " + str(path))
    require(counts == sorted(set(counts)), "Invalid execution order " + str(path))
    png = 0
    for c in cells:
        source = c.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        ast.parse(source)
        for out in c.get("outputs", []):
            require(out.get("output_type") != "error", "Notebook error " + str(path))
            if "image/png" in out.get("data", {}):
                b = out["data"]["image/png"]
                b = "".join(b) if isinstance(b, list) else b
                png_check(base64.b64decode(b, validate=False))
                png += 1
    require(png >= 1, "Missing saved PNG " + str(path))
    return {"file": path.name, "execution_counts": counts, "png_outputs": png}

def verify(root=ROOT):
    root = Path(root)
    required = [
        "pyproject.toml", "requirements-test.txt", "REPRODUCE.md",
        "reproducibility/external_assets.json",
        "scripts/run_ci.py", "scripts/check_showcase.py",
        "scripts/reproduce_final_evidence.py",
        "src/biohub_tracking/__init__.py",
        "reports/portfolio_summary.json",
    ]
    for rel in required:
        require((root / rel).is_file(), "Missing required archive file " + rel)

    files = 0
    py_files = 0
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if any(part in IGNORED for part in rel.parts):
            continue
        require(not p.is_symlink(), "Symlink " + str(rel))
        if not p.is_file():
            continue
        files += 1
        name = rel.as_posix()
        check_path(name)
        data = p.read_bytes()
        require(len(data) <= 4_000_000, "Oversized tracked file " + name)
        if p.suffix == ".png":
            png_check(data)
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("Unexpected binary text file " + name)
        if name != "scripts/verify_portfolio.py":
            require(not SECRET.search(text), "Credential-like value in " + name)
        if p.suffix == ".py":
            ast.parse(text, filename=name)
            py_files += 1

    evidence_notebooks = [
        p for p in sorted((root / "notebooks").glob("*.ipynb"))
        if p.name != "portfolio.ipynb"
    ]
    require(len(evidence_notebooks) == 7, "Expected seven executed evidence notebooks")
    notebooks = [check_notebook(p) for p in evidence_notebooks]

    summary = json.loads((root / "reports/portfolio_summary.json").read_text())
    require(summary["official_public_score"] == 0.947, "Official evidence contract changed")

    assets = json.loads((root / "reproducibility/external_assets.json").read_text())
    require(assets["organizer"]["commit"] == "075fc5f5a52d11077f9dc2b074644618f26939e2", "Organizer pin drift")
    require(assets["public_0953_lineage"]["expected_output_sha256"].startswith("d52a5d"), "Public 0.953 hash drift")

    return {
        "status": "passed",
        "tracked_files_checked": files,
        "python_files_parsed": py_files,
        "executed_evidence_notebooks": notebooks,
        "official_public_score": summary["official_public_score"],
    }

if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
