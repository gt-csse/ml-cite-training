"""Reproduce ML-cite training windows from shared full-document annotations.

Only the standard library is needed. Prepared data is read solely for the final
byte comparison; it never supplies conversion inputs. Verification never writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

SPLITS = ("train", "dev", "test")
FIELDS = ("case_name", "court", "date", "pin_cite")
KINDS = ("FullCaseCitation", "ShortCaseCitation", "DocketCitation")
DOCUMENT_SCHEMA = "mellea-lrc/annotation-document/v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(raw: bytes) -> object:
    def invalid(value: str) -> None:
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=invalid)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _path(release: Path, name: str) -> Path:
    _require(isinstance(name, str) and bool(name), "Missing package-relative path")
    relative = PurePosixPath(name)
    _require(not relative.is_absolute() and "\\" not in name
             and all(part not in {"", ".", ".."} for part in name.split("/")),
             f"Unsafe package-relative path: {name}")
    result = release.joinpath(*relative.parts)
    _require(result.resolve().is_relative_to(release) and not result.is_symlink(),
             f"Escaping or symlinked input: {name}")
    return result


def _bound(release: Path, manifest: dict, name: str) -> bytes:
    raw = _path(release, name).read_bytes()
    _require(manifest["files"].get(name) == {"bytes": len(raw), "sha256": _sha(raw)},
             f"Input hash/length differs from manifest: {name}")
    return raw


def _span(value: object, text: str, label: str) -> dict | None:
    if value is None:
        return None
    _require(isinstance(value, dict) and isinstance(value.get("source"), dict),
             f"{label}: expected a canonical source field or null")
    source = value["source"]
    kind = source.get("kind")
    _require(kind in {"quoted", "inferred", "not_stated", "not_applicable"},
             f"{label}: unknown source kind")
    if kind != "quoted":
        _require(not {"start", "end", "quote"}.intersection(source),
                 f"{label}: unquoted source contains quotation attributes")
        return None
    start, end, quote = (source.get(key) for key in ("start", "end", "quote"))
    _require(type(start) is int and type(end) is int and isinstance(quote, str)
             and 0 <= start < end <= len(text) and text[start:end] == quote,
             f"{label}: quotation does not exactly re-slice embedded text")
    return {"start": start, "end": end, "quote": quote}


def _coverage(rows: list[dict]) -> dict:
    return {field: sum(row["targets"][field] is not None for row in rows) for field in FIELDS}


def convert(release: Path, output_dir: Path | None = None) -> dict:
    """Verify exact reconstruction, optionally writing three files outside release."""
    release = Path(release).resolve()
    _require(release.is_dir(), "Release directory does not exist")
    if output_dir is not None:
        output_dir = Path(output_dir)
        _require(not output_dir.exists() and not output_dir.is_symlink(), "Output path already exists")
        output_dir = output_dir.resolve()
        _require(not output_dir.is_relative_to(release), "Output must be outside the release")

    manifest = _json((release / "manifest.json").read_bytes())
    _require(isinstance(manifest, dict) and manifest.get("schema_version") == 2
             and manifest.get("release_kind") == "natural-only", "Unsupported release manifest")
    conversion = manifest["conversion"]
    _require(conversion.get("module") == "case_name_span.build_multicomponent"
             and conversion.get("left_chars") == 384 and conversion.get("right_chars") == 128
             and conversion.get("citation_kinds") == list(KINDS), "Unsupported frozen conversion recipe")
    _require(manifest.get("components") == list(FIELDS), "Wrong target field inventory")
    _require(manifest.get("split_files") == {split: f"data/{split}.jsonl" for split in SPLITS},
             "Wrong prepared split paths")
    source_set = manifest["source_set"]
    _require(isinstance(source_set, str) and bool(source_set), "Missing source-set label")
    assignments = conversion["split_assignments"]
    _require(isinstance(assignments, dict) and assignments == manifest["document_splits"]
             and all(value in SPLITS for value in assignments.values()), "Wrong document split assignments")
    _require(manifest["group_splits"] == {f"doc:{Path(name).stem}": split
                                         for name, split in assignments.items()}, "Wrong document grouping")

    index = _json(_bound(release, manifest, "annotation-index.json"))
    _require(isinstance(index, dict) and index.get("schema") == "mellea-lrc/annotation-index/v1"
             and isinstance(index.get("documents"), list), "Malformed annotation index")
    sources = manifest["sources"]
    source_by_document = {}
    for source in sources:
        document = PurePosixPath(source["text"]).name
        _require(document not in source_by_document and document.endswith(".txt"), "Duplicate source document")
        stem = Path(document).stem
        _require(source["annotation"] == f"{conversion['documents_dir']}/{stem}.jsonl"
                 and source["text"] == f"{conversion['texts_dir']}/{document}", "Wrong source paths")
        for key, hash_key in (("annotation", "annotation_sha256"), ("text", "text_sha256")):
            evidence = manifest["files"][source[key]]
            _require(evidence["sha256"] == source[hash_key]
                     and index["assets"].get(source[key]) == evidence, "Source evidence/index mismatch")
        source_by_document[document] = source
    _require(set(index["assets"]) == {source[key] for source in sources for key in ("annotation", "text")},
             "Wrong indexed source-asset inventory")
    _require(set(assignments) == set(source_by_document), "Split/source document inventories differ")

    rows = {split: [] for split in SPLITS}
    seen_ids, documents, paths = set(), set(), set()
    for item in index["documents"]:
        name = item["path"]
        _require(name not in paths, "Duplicate indexed annotation path")
        paths.add(name)
        raw = _bound(release, manifest, name)
        _require(_sha(raw) == item["sha256"], f"Indexed annotation hash mismatch: {name}")
        document = _json(raw)
        _require(isinstance(document, dict) and document.get("schema") == DOCUMENT_SCHEMA
                 and isinstance(document.get("text"), str) and isinstance(document.get("rows"), list)
                 and isinstance(document.get("header"), dict), f"Malformed annotation document: {name}")
        header, text = document["header"], document["text"]
        filename = header.get("document")
        _require(filename in assignments and filename not in documents and item["id"] == filename
                 and name == f"annotations/{Path(filename).stem}.json", f"Wrong document identity: {name}")
        documents.add(filename)
        source = source_by_document[filename]
        text_info = header.get("text", {})
        _require(header.get("unit") == "header" and header.get("dataset") == source["header_dataset"]
                 and text_info.get("path") == source["header_text_path"]
                 and len(text) == text_info.get("length") == item["length"] == source["text_length"]
                 and _sha(text.encode("utf-8")) == text_info.get("sha256") == item["text_sha256"] == source["text_sha256"],
                 f"Embedded source text/header mismatch: {name}")
        citations, roots = 0, 0
        for row in document["rows"]:
            _require(isinstance(row, dict) and isinstance(row.get("unit"), str)
                     and row["unit"] != "header", f"Malformed annotation row: {name}")
            if row["unit"] != "citation":
                continue
            identifier = row.get("id")
            _require(isinstance(identifier, str) and bool(identifier) and identifier not in seen_ids,
                     f"Missing or duplicate citation ID: {identifier!r}")
            seen_ids.add(identifier)
            citations += 1
            _require(type(row.get("is_root")) is bool and "root_id" in row
                     and (row["root_id"] is None or isinstance(row["root_id"], str)),
                     f"{identifier}: malformed root metadata")
            roots += row["is_root"]
            _require(isinstance(row.get("kind"), str) and bool(row["kind"]), f"{identifier}: missing kind")
            if row["kind"] not in KINDS:
                continue
            _require(all(field in row for field in ("locator", *FIELDS)),
                     f"{identifier}: missing explicit source field; cannot infer absence")
            locator = _span(row["locator"], text, f"{identifier}.locator")
            targets = {field: _span(row[field], text, f"{identifier}.{field}") for field in FIELDS}
            if locator is None or not any(value is not None for value in targets.values()):
                continue
            spans = [locator, *(value for value in targets.values() if value is not None)]
            start = max(0, min(min(span["start"] for span in spans), locator["start"] - 384))
            end = min(len(text), max(span["end"] for span in spans) + 128)

            def local(span: dict | None) -> dict | None:
                return None if span is None else {"start": span["start"] - start,
                                                  "end": span["end"] - start, "quote": span["quote"]}

            example = {"id": identifier, "source_set": source_set, "document": filename,
                       "case_group": f"doc:{Path(filename).stem}", "citation_id": identifier,
                       "citation_kind": row["kind"], "context_text": text[start:end],
                       "context_start": start, "context_end": end, "locator": local(locator),
                       "targets": {field: local(span) for field, span in targets.items()}}
            example["case_name"] = example["targets"]["case_name"]
            rows[assignments[filename]].append(example)
        _require(citations == item["citations"] and roots == item["roots"], f"Indexed counts differ: {name}")
    _require(documents == set(assignments) and paths == {path.relative_to(release).as_posix()
             for path in (release / "annotations").iterdir()}, "Wrong annotation document inventory")
    _require(manifest["document_annotations"]["documents"] == len(documents)
             and manifest["document_annotations"]["citations"] == len(seen_ids), "Wrong annotation totals")

    review_metadata = manifest["review"]
    _require(review_metadata["path"] == "exclusions/review.json", "Wrong exclusion ledger path")
    review = _json(_bound(release, manifest, review_metadata["path"]))
    _require(isinstance(review, dict) and review.get("scope") == "natural_all_splits"
             and review_metadata.get("scope") == "natural_all_splits"
             and isinstance(review.get("exclusions"), list), "Malformed rendering-exclusion ledger")
    exclusions = {}
    for entry in review["exclusions"]:
        identifier, split = entry.get("id"), entry.get("split")
        _require(isinstance(identifier, str) and identifier not in exclusions and split in SPLITS
                 and entry.get("exclusion_kind") == "annotation_rendering_exception"
                 and isinstance(entry.get("fields"), list) and bool(entry["fields"])
                 and set(entry["fields"]) <= set(FIELDS) and bool(entry.get("reason") or entry.get("reasons")),
                 "Malformed or duplicate rendering exclusion")
        exclusions[identifier] = split
    removed_ids = {split: sorted(key for key, value in exclusions.items() if value == split) for split in SPLITS}
    _require(len(exclusions) == review.get("count") == review_metadata["count"]
             and removed_ids == review_metadata["removed_ids"]
             and {split: len(ids) for split, ids in removed_ids.items()} == review_metadata["removed_rows"],
             "Exclusion ledger and manifest disagree")

    payloads, source_hashes, removed_hashes, retained = {}, {}, {}, {}
    for split in SPLITS:
        rows[split].sort(key=lambda row: row["id"])
        raw_lines, kept_lines, kept_rows = [], [], []
        for row in rows[split]:
            line = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            raw_lines.append(line)
            if row["id"] in exclusions:
                _require(exclusions[row["id"]] == split, "Excluded ID belongs to a different split")
                removed_hashes[row["id"]] = _sha(line)
            else:
                kept_lines.append(line)
                kept_rows.append(row)
        source_hashes[split] = _sha(b"".join(raw_lines))
        payloads[split], retained[split] = b"".join(kept_lines), kept_rows
    _require(source_hashes == manifest["source_sha256"]
             and {split: len(values) for split, values in rows.items()} == manifest["source_rows"]
             and {split: _coverage(values) for split, values in rows.items()} == manifest["source_component_coverage"],
             "Raw source reconstruction differs from frozen hashes/counts/coverage")
    _require(removed_hashes == review_metadata["removed_line_sha256"], "Excluded IDs or exact line hashes differ")
    counts = {split: len(values) for split, values in retained.items()}
    hashes = {split: _sha(payload) for split, payload in payloads.items()}
    _require(counts == manifest["rows"] and {split: _coverage(values) for split, values in retained.items()}
             == manifest["component_coverage"], "Retained row counts/coverage differ")
    for split, payload in payloads.items():
        expected = _bound(release, manifest, manifest["split_files"][split])
        _require(payload == expected, f"{split}: reconstructed bytes differ from prepared data")

    if output_dir is not None:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir()
        for split, payload in payloads.items():
            (output_dir / f"{split}.jsonl").write_bytes(payload)
    return {"status": "byte-identical", "byte_identical": True, "documents": len(documents),
            "source_rows": manifest["source_rows"], "rows": counts, "sha256": hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true", help="Verify without writing")
    mode.add_argument("--output-dir", type=Path, help="Write only after exact verification, to a new directory")
    args = parser.parse_args()
    try:
        result = convert(args.release, args.output_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Conversion failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
