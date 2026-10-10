"""Fresh-clone integrity checks for the reproducible Biohub archive."""
from __future__ import annotations
import ast, base64, binascii, hashlib, json, re, struct, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", ".venv", ".venv-review"}
PRIVATE_PARTS = {".aws", ".kaggle", "data", "outputs", "artifacts", "checkpoints", "weights", "cache", "caches", "external", "repro_data"}
BAD_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".pkl", ".pickle", ".npz", ".npy", ".parquet", ".pyz", ".zip", ".gz", ".tar"}
SECRET = re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
PRIVATE_LOCATOR = re.compile(r"(?<![\w:])/(?:home|workspace|mnt)/|s3://|arn:aws(?:-[a-z]+)?:", re.IGNORECASE)
COMPARATOR_KEYS = {
    "leader_snapshot", "leader_snapshot_score", "leader_snapshot_observed_utc",
    "retrieved_public_leader_score", "gap_to_retrieved_public_leader",
    "leaderboard_observed_utc", "fresh_leaderboard_query", "official_gap_snapshot",
    "recorded_gap", "independently_recreated_leader_training",
}

def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_public_metadata(value, location="public metadata"):
    """Reject operational locations and ranking metadata, retaining scientific gaps."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            require(normalized not in COMPARATOR_KEYS and "leaderboard" not in normalized,
                    "Comparator metadata in " + location + ": " + str(key))
            check_public_metadata(item, location + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_public_metadata(item, f"{location}[{index}]")
    elif isinstance(value, str):
        require(not PRIVATE_LOCATOR.search(value), "Private operational location in " + location)


def check_publication_receipt(root):
    """Authenticate transformed publication copies without rewriting source hashes."""
    root = Path(root)
    receipt = json.loads((root / "reports/publication_redaction.json").read_text())
    require(receipt.get("reexecuted_research") is False, "Publication must not claim research reexecution")
    changed = receipt.get("changed_files", [])
    require(bool(changed), "Publication transformation receipt is empty")
    seen = set()
    for item in changed:
        rel = item["path"]
        check_path(rel)
        require(rel not in seen, "Duplicate publication receipt path " + rel)
        seen.add(rel)
        path = root / rel
        require(path.is_file() and not path.is_symlink(), "Missing publication copy " + rel)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item["published_sha256"],
                "Published copy hash mismatch " + rel)
    for name, expected in receipt.get("notebook_outputs_unchanged", {}).items():
        require(Path(name).name == name and name.endswith(".ipynb"), "Invalid notebook receipt path")
        notebook = json.loads((root / "notebooks" / name).read_text())
        output_bytes = json.dumps([cell.get("outputs") for cell in notebook["cells"]], sort_keys=True).encode()
        require(hashlib.sha256(output_bytes).hexdigest() == expected, "Preserved notebook output hash mismatch " + name)
    return len(changed)

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
        if any(part in IGNORED or part.endswith(".egg-info") for part in rel.parts):
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
        if p.suffix in {".json", ".ipynb"} and rel.parts[0] in {"reports", "notebooks", "reproducibility", "public-demo"}:
            check_public_metadata(json.loads(text), name)

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

    published_copies = check_publication_receipt(root)

    return {
        "status": "passed",
        "tracked_files_checked": files,
        "python_files_parsed": py_files,
        "executed_evidence_notebooks": notebooks,
        "official_public_score": summary["official_public_score"],
        "authenticated_publication_copies": published_copies,
    }

if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
