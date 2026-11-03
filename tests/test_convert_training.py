"""Exercise conversion with hand-specified rows, independently of the builder."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.convert_training import convert


FIELDS = ("case_name", "court", "date", "pin_cite")
SPLITS = ("train", "dev", "test")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                   for row in rows).encode("utf-8")


def quoted(text, quote):
    start = text.index(quote)
    return {"source": {"kind": "quoted", "start": start,
                       "end": start + len(quote), "quote": quote}}


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def seal(release):
    """Refresh transport checksums after deliberately changing fixture evidence."""
    index_path = release / "annotation-index.json"
    index = json.loads(index_path.read_bytes())
    for item in index["documents"]:
        item["sha256"] = sha((release / item["path"]).read_bytes())
    index_path.write_bytes(json_bytes(index))
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"] = {name: {"bytes": len(raw), "sha256": sha(raw)}
                         for name, raw in snapshot(release).items() if name != "manifest.json"}
    manifest_path.write_bytes(json_bytes(manifest))


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "release"
    for folder in ("annotations", "data", "exclusions", "sources/fixture/documents",
                   "sources/fixture/texts"):
        (root / folder).mkdir(parents=True)

    a_text = "pré\r\nAlpha v. Beta, 1 F.3d 2, 4 (X 2000).\r\nfin"
    b_text = "\r\nNo. 12, été."
    c_text = "z" * 400 + "\r\nGamma, 2 U.S. 3 (1999)." + "q" * 140

    a = {"unit": "citation", "id": "a-o001", "is_root": True,
         "root_id": "a-o001", "kind": "FullCaseCitation",
         "locator": quoted(a_text, "1 F.3d 2"),
         "case_name": quoted(a_text, "Alpha v. Beta"), "court": quoted(a_text, "X"),
         "date": quoted(a_text, "2000"), "pin_cite": quoted(a_text, "4")}
    # Normalization metadata must not replace the literal source quotation.
    a["case_name"]["normalization"] = {"kind": "value", "value": "ALPHA"}

    def variant(identifier, **changes):
        return {**copy.deepcopy(a), "id": identifier, "root_id": identifier, **changes}

    a_excluded = variant("a-o002")
    a_short = variant("a-o003", kind="ShortCaseCitation",
                      case_name={"source": {"kind": "inferred"}},
                      court={"source": {"kind": "not_stated"}}, date=None)
    a_other_kind = variant("a-o004", kind="FullLawCitation")
    a_no_targets = variant("a-o005", **dict.fromkeys(FIELDS))
    a_no_locator = variant("a-o006", locator={"source": {"kind": "not_applicable"}})
    b = {"unit": "citation", "id": "b-o001", "is_root": True,
         "root_id": "b-o001", "kind": "DocketCitation", "locator": quoted(b_text, "No. 12"),
         "case_name": quoted(b_text, "été"), "court": {"source": {"kind": "inferred"}},
         "date": None, "pin_cite": None}
    c = {"unit": "citation", "id": "c-o001", "is_root": True,
         "root_id": "c-o001", "kind": "FullCaseCitation", "locator": quoted(c_text, "2 U.S. 3"),
         "case_name": quoted(c_text, "Gamma"), "court": None,
         "date": quoted(c_text, "1999"), "pin_cite": None}

    # Expected offsets and context limits are specified by hand, not generated
    # by the converter or by the archived builder.
    full = {"id": "a-o001", "source_set": "fixture", "document": "a.txt",
            "case_group": "doc:a", "citation_id": "a-o001", "citation_kind": "FullCaseCitation",
            "context_text": a_text, "context_start": 0, "context_end": 46,
            "locator": {"start": 20, "end": 28, "quote": "1 F.3d 2"},
            "targets": {"case_name": {"start": 5, "end": 18, "quote": "Alpha v. Beta"},
                        "court": {"start": 33, "end": 34, "quote": "X"},
                        "date": {"start": 35, "end": 39, "quote": "2000"},
                        "pin_cite": {"start": 30, "end": 31, "quote": "4"}},
            "case_name": {"start": 5, "end": 18, "quote": "Alpha v. Beta"}}
    excluded = {**copy.deepcopy(full), "id": "a-o002", "citation_id": "a-o002"}
    short = {**copy.deepcopy(full), "id": "a-o003", "citation_id": "a-o003",
             "citation_kind": "ShortCaseCitation",
             "targets": {"case_name": None, "court": None, "date": None,
                         "pin_cite": {"start": 30, "end": 31, "quote": "4"}}, "case_name": None}
    docket = {"id": "b-o001", "source_set": "fixture", "document": "b.txt",
              "case_group": "doc:b", "citation_id": "b-o001", "citation_kind": "DocketCitation",
              "context_text": b_text, "context_start": 0, "context_end": 14,
              "locator": {"start": 2, "end": 8, "quote": "No. 12"},
              "targets": {"case_name": {"start": 10, "end": 13, "quote": "été"},
                          "court": None, "date": None, "pin_cite": None},
              "case_name": {"start": 10, "end": 13, "quote": "été"}}
    windowed = {"id": "c-o001", "source_set": "fixture", "document": "c.txt",
                "case_group": "doc:c", "citation_id": "c-o001", "citation_kind": "FullCaseCitation",
                "context_text": "z" * 375 + "\r\nGamma, 2 U.S. 3 (1999)." + "q" * 126,
                "context_start": 25, "context_end": 551,
                "locator": {"start": 384, "end": 392, "quote": "2 U.S. 3"},
                "targets": {"case_name": {"start": 377, "end": 382, "quote": "Gamma"},
                            "court": None, "date": {"start": 394, "end": 398, "quote": "1999"},
                            "pin_cite": None},
                "case_name": {"start": 377, "end": 382, "quote": "Gamma"}}
    expected = {"train": [full, short], "dev": [docket], "test": [windowed]}
    raw_rows = {"train": [full, excluded, short], "dev": [docket], "test": [windowed]}

    documents, assets, sources = [], {}, []
    for stem, text, rows in [("c", c_text, [c]), ("b", b_text, [b]),
                             ("a", a_text, [a_no_locator, a_short, a_no_targets, a_excluded,
                                            a_other_kind, a, {"unit": "note", "text": "retained"}])]:
        citations = [row for row in rows if row["unit"] == "citation"]
        text_raw = text.encode("utf-8")
        header = {"unit": "header", "dataset": "fixture", "document": f"{stem}.txt",
                  "text": {"path": f"fixture/{stem}.txt", "sha256": sha(text_raw), "length": len(text)}}
        annotation_path = f"annotations/{stem}.json"
        document_raw = json_bytes({"schema": "mellea-lrc/annotation-document/v1",
                                   "header": header, "text": text, "rows": rows})
        (root / annotation_path).write_bytes(document_raw)
        documents.append({"id": f"{stem}.txt", "path": annotation_path, "sha256": sha(document_raw),
                          "text_sha256": sha(text_raw), "length": len(text),
                          "citations": len(citations), "roots": len(citations)})
        annotation_asset = f"sources/fixture/documents/{stem}.jsonl"
        text_asset = f"sources/fixture/texts/{stem}.txt"
        for name, payload in ((annotation_asset, jsonl([header, *rows])), (text_asset, text_raw)):
            (root / name).write_bytes(payload)
            assets[name] = {"bytes": len(payload), "sha256": sha(payload)}
        sources.append({"annotation": annotation_asset, "annotation_sha256": assets[annotation_asset]["sha256"],
                        "text": text_asset, "text_sha256": sha(text_raw), "text_length": len(text),
                        "header_dataset": "fixture", "header_text_path": f"fixture/{stem}.txt"})

    (root / "annotation-index.json").write_bytes(json_bytes({
        "schema": "mellea-lrc/annotation-index/v1", "id": "fixture", "documents": documents, "assets": assets}))
    (root / "exclusions/review.json").write_bytes(json_bytes({
        "scope": "natural_all_splits", "count": 1, "exclusions": [{
            "id": "a-o002", "split": "train", "fields": ["case_name"],
            "reason": "Reviewed rendering exception; preserve the source annotation.",
            "exclusion_kind": "annotation_rendering_exception"}]}))
    for split in SPLITS:
        (root / f"data/{split}.jsonl").write_bytes(jsonl(expected[split]))
    assignments = {"a.txt": "train", "b.txt": "dev", "c.txt": "test"}
    source_coverage = {"train": dict(zip(FIELDS, [2, 2, 2, 3])),
                       "dev": dict(zip(FIELDS, [1, 0, 0, 0])), "test": dict(zip(FIELDS, [1, 0, 1, 0]))}
    coverage = {**source_coverage, "train": dict(zip(FIELDS, [1, 1, 1, 2]))}
    manifest = {"schema_version": 2, "release_kind": "natural-only", "source_set": "fixture",
                "components": list(FIELDS), "split_files": {s: f"data/{s}.jsonl" for s in SPLITS},
                "conversion": {"module": "case_name_span.build_multicomponent", "left_chars": 384,
                               "right_chars": 128, "seed": 42, "citation_kinds": [
                                   "FullCaseCitation", "ShortCaseCitation", "DocketCitation"],
                               "documents_dir": "sources/fixture/documents", "texts_dir": "sources/fixture/texts",
                               "split_assignments": assignments},
                "document_splits": assignments, "group_splits": {"doc:a": "train", "doc:b": "dev", "doc:c": "test"},
                "document_annotations": {"documents": 3, "citations": 8}, "sources": sources,
                "source_rows": {"train": 3, "dev": 1, "test": 1}, "rows": {"train": 2, "dev": 1, "test": 1},
                "source_sha256": {s: sha(jsonl(raw_rows[s])) for s in SPLITS},
                "source_component_coverage": source_coverage, "component_coverage": coverage,
                "review": {"path": "exclusions/review.json", "scope": "natural_all_splits", "count": 1,
                           "removed_ids": {"train": ["a-o002"], "dev": [], "test": []},
                           "removed_rows": {"train": 1, "dev": 0, "test": 0},
                           "removed_line_sha256": {"a-o002": sha(jsonl([excluded]))}}}
    (root / "manifest.json").write_bytes(json_bytes(manifest))
    seal(root)
    return root, {split: jsonl(rows) for split, rows in expected.items()}


def test_verify_and_export_exact_bytes_without_mutating_release(release, tmp_path):
    root, expected = release
    before = snapshot(root)
    verified = convert(root)
    assert verified["byte_identical"] is True
    assert verified["rows"] == {"train": 2, "dev": 1, "test": 1}
    assert verified["source_rows"] == {"train": 3, "dev": 1, "test": 1}
    output = tmp_path / "fresh" / "export"
    assert convert(root, output) == verified
    assert snapshot(output) == {f"{split}.jsonl": raw for split, raw in expected.items()}
    assert convert(root) == verified
    assert snapshot(root) == before


def test_reconstruction_rejects_changed_prepared_bytes_even_with_new_hash(release, tmp_path):
    root, _ = release
    prepared = root / "data/train.jsonl"
    prepared.write_bytes(b" " + prepared.read_bytes())  # Valid JSON, different line bytes.
    seal(root)
    before = snapshot(root)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="reconstructed bytes differ"):
        convert(root, output)
    assert not output.exists()
    assert snapshot(root) == before


@pytest.mark.parametrize("corruption", ["quote", "embedded_text", "split", "review_split"])
def test_invalid_evidence_fails_before_creating_output(release, tmp_path, corruption):
    root, _ = release
    if corruption in {"quote", "embedded_text"}:
        path = root / "annotations/a.json"
        document = json.loads(path.read_bytes())
        if corruption == "quote":
            next(row for row in document["rows"] if row.get("id") == "a-o001")["case_name"]["source"]["quote"] = "wrong"
        else:
            document["text"] = document["text"].replace("pré", "pre")
        path.write_bytes(json_bytes(document))
    elif corruption == "split":
        path = root / "manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["document_splits"]["a.txt"] = "test"
        path.write_bytes(json_bytes(manifest))
    else:
        path = root / "exclusions/review.json"
        review = json.loads(path.read_bytes())
        review["exclusions"][0]["split"] = "test"
        path.write_bytes(json_bytes(review))
    seal(root)  # Semantic checks must still catch consistently rehashed evidence.
    before = snapshot(root)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        convert(root, output)
    assert not output.exists()
    assert snapshot(root) == before


def test_output_cannot_overwrite_or_write_inside_release(release, tmp_path):
    root, _ = release
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "train.jsonl").write_bytes(b"keep me\n")
    before = snapshot(root)
    with pytest.raises(ValueError, match="already exists"):
        convert(root, existing)
    assert snapshot(existing) == {"train.jsonl": b"keep me\n"}
    with pytest.raises(ValueError, match="outside the release"):
        convert(root, root / "new-output")
    assert snapshot(root) == before
    assert not (root / "new-output").exists()


def test_cli_requires_exactly_one_mode(release):
    root, _ = release
    script = Path(__file__).resolve().parents[1] / "tools/convert_training.py"
    common = [sys.executable, str(script), "--release", str(root)]
    for extra in ([], ["--verify", "--output-dir", str(root.parent / "unused")]):
        result = subprocess.run([*common, *extra], capture_output=True, text=True)
        assert result.returncode == 2
    result = subprocess.run([*common, "--verify"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["byte_identical"] is True
