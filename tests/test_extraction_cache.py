from pathlib import Path

import smartfileman.scanner as scanner


def test_scan_reuses_extracted_units_by_content_hash(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("motion and force", encoding="utf-8")
    cache = tmp_path / ".smartfile" / "extraction-cache.json"
    original = scanner.extract_units
    calls = 0

    def counted_extract(path: Path, max_chunk_chars: int):
        nonlocal calls
        calls += 1
        return original(path, max_chunk_chars=max_chunk_chars)

    monkeypatch.setattr(scanner, "extract_units", counted_extract)

    first = scanner.scan(tmp_path, extraction_cache_path=cache)
    second = scanner.scan(tmp_path, extraction_cache_path=cache)

    assert calls == 1
    assert first.ready_files[0].units == second.ready_files[0].units


def test_scan_invalidates_extraction_cache_when_content_changes(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("motion and force", encoding="utf-8")
    cache = tmp_path / ".smartfile" / "extraction-cache.json"
    original = scanner.extract_units
    calls = 0

    def counted_extract(path: Path, max_chunk_chars: int):
        nonlocal calls
        calls += 1
        return original(path, max_chunk_chars=max_chunk_chars)

    monkeypatch.setattr(scanner, "extract_units", counted_extract)

    scanner.scan(tmp_path, extraction_cache_path=cache)
    source.write_text("motion and anime", encoding="utf-8")
    result = scanner.scan(tmp_path, extraction_cache_path=cache)

    assert calls == 2
    assert result.ready_files[0].units[0].text == "motion and anime"
