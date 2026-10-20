from pathlib import Path

import pytest

from smartfileman.config import Settings
from smartfileman.operations import OperationError
from smartfileman.paths import UnsafePathError, safe_destination
from smartfileman.readers import READERS
from smartfileman.scanner import content_hash, scan
from smartfileman.service import apply, make_preview, undo


def settings_for(root: Path) -> Settings:
    return Settings(
        data_dir=Path(".smartfile"),
        allowed_roots=str(root),
        ollama_host="http://127.0.0.1:11434",
        ollama_model="embeddinggemma-2:740m",
    )


def test_reader_registry_selects_by_extension() -> None:
    assert ".txt" in READERS
    assert ".pdf" in READERS
    assert READERS[".txt"] is not READERS[".pdf"]


def test_preview_apply_and_undo(tmp_path: Path) -> None:
    source = tmp_path / "physics_notes.txt"
    source.write_text("Newton motion and force", encoding="utf-8")
    settings = settings_for(tmp_path)

    plan = make_preview(tmp_path, mode="fuzzy", settings=settings)
    operation = plan.operations[0]
    assert operation.subject == "Force - Motion - Newton"
    assert operation.status == "ready"
    assert source.exists()

    results = apply(tmp_path, plan.plan_id, {operation.operation_id}, True, settings)
    assert results[0]["status"] == "moved"
    moved = tmp_path / operation.destination_relative
    assert moved.read_text(encoding="utf-8") == "Newton motion and force"
    assert not source.exists()

    undo(tmp_path, results[0]["move_id"], settings)
    assert source.read_text(encoding="utf-8") == "Newton motion and force"
    assert not moved.exists()


def test_preview_does_not_move_files(tmp_path: Path) -> None:
    source = tmp_path / "biology_notes.txt"
    source.write_text("cell genetics", encoding="utf-8")
    before = content_hash(source)

    plan = make_preview(tmp_path, settings=settings_for(tmp_path))

    assert source.exists()
    assert content_hash(source) == before
    assert not (tmp_path / "_organized").exists()
    assert (tmp_path / ".smartfile" / "plans" / f"{plan.plan_id}.json").exists()


def test_stale_source_hash_blocks_apply(tmp_path: Path) -> None:
    source = tmp_path / "math_notes.txt"
    source.write_text("algebra", encoding="utf-8")
    settings = settings_for(tmp_path)
    plan = make_preview(tmp_path, settings=settings)
    source.write_text("algebra changed", encoding="utf-8")

    with pytest.raises(OperationError, match="Source changed"):
        apply(tmp_path, plan.plan_id, {plan.operations[0].operation_id}, True, settings)


def test_symlink_is_ignored(tmp_path: Path) -> None:
    source = tmp_path / "physics.txt"
    source.write_text("motion", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(source)

    result = scan(tmp_path)

    assert [file.filename for file in result.files] == ["physics.txt"]


def test_destination_rejects_symlinked_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafePathError, match="symlink"):
        safe_destination(tmp_path, "link/file.txt")


def test_plan_id_changes_when_mode_changes(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("motion and force", encoding="utf-8")
    settings = settings_for(tmp_path)

    fuzzy = make_preview(tmp_path, mode="fuzzy", settings=settings)
    semantic = make_preview(tmp_path, mode="semantic", provider_kind="fake", settings=settings)

    assert fuzzy.plan_id != semantic.plan_id
