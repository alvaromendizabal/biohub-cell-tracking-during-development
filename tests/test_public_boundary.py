"""Publication boundaries reject leaks without rejecting real tracking concepts."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_showcase import check_local_links, check_svg
from verify_portfolio import check_public_metadata, check_publication_receipt


@pytest.mark.parametrize("key", ["leader_snapshot", "official_gap_snapshot", "leaderboard_observed_utc", "retrieved_public_leader_score"])
def test_nested_comparator_metadata_is_rejected(key):
    with pytest.raises(ValueError, match="Comparator"):
        check_public_metadata({"results": [{key: 0.123}]})


@pytest.mark.parametrize("location", ["/home/example/private/file", "/workspace/private/image", "/mnt/input/data", "s3://example-bucket/input", "arn:aws:s3:::example-bucket"])
def test_operational_locations_are_rejected(location):
    with pytest.raises(ValueError, match="Private operational"):
        check_public_metadata({"receipts": [{"source": location}]})


def test_scientific_gap_counters_and_own_results_are_preserved():
    check_public_metadata({"deepcenter_gap_accepted": 17, "gap2_added_edges": 213,
                           "official_public_score": 0.947, "candidate": 0.946,
                           "source": "https://github.com/example/public-source"})


def test_relative_navigation_cannot_escape_public_repository(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/valid.md").write_text("public")
    check_local_links(tmp_path, "README.md", "[Valid](docs/valid.md)")
    with pytest.raises(ValueError, match="escapes"):
        check_local_links(tmp_path, "README.md", "[Invalid](../private.md)")


def test_publication_copy_hash_detects_later_mutation(tmp_path):
    (tmp_path / "reports").mkdir()
    copy = tmp_path / "reports/result.json"
    copy.write_text('{"own_score":0.947}')
    receipt = {"reexecuted_research": False, "changed_files": [
        {"path": "reports/result.json", "published_sha256": hashlib.sha256(copy.read_bytes()).hexdigest()}]}
    (tmp_path / "reports/publication_redaction.json").write_text(json.dumps(receipt))
    assert check_publication_receipt(tmp_path) == 1
    copy.write_text('{"own_score":0.999}')
    with pytest.raises(ValueError, match="hash mismatch"):
        check_publication_receipt(tmp_path)


def test_native_svg_remains_self_contained(tmp_path):
    path = tmp_path / "asset.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" role="img"><title>Title</title><desc>Description</desc><image href="https://example.org/image.png"/></svg>')
    with pytest.raises(ValueError, match="External SVG"):
        check_svg(path)
