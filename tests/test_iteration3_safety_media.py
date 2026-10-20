from pathlib import Path

import pytest

import smartfileman.operations as operations
from smartfileman.config import Settings
from smartfileman.embeddings import FakeEmbeddingProvider
from smartfileman.indexer import index_scan
from smartfileman.operations import OperationError, apply_plan, save_plan
from smartfileman.readers import normalize_media_filename
from smartfileman.schemas import Plan, PlanOperation
from smartfileman.scanner import scan
from smartfileman.service import make_preview
from smartfileman.cli import _select_operations


def settings_for(root: Path, data_dir: str = ".smartfile") -> Settings:
    return Settings(_env_file=None, data_dir=Path(data_dir), allowed_roots=str(root))


def test_unknown_model_identity_does_not_reuse_persistent_embeddings(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("motion and force", encoding="utf-8")
    database = tmp_path / ".smartfile" / "index.sqlite3"

    first = FakeEmbeddingProvider()
    first.model_digest = "unknown"
    index_scan(scan(tmp_path), database, first)

    second = FakeEmbeddingProvider()
    second.model_digest = "unknown"
    result = index_scan(scan(tmp_path), database, second)

    assert result["embedded"] == 1
    assert result["reused"] == 0
    assert second.calls == 1


def test_custom_storage_directory_is_not_scanned(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("motion and force", encoding="utf-8")
    settings = settings_for(tmp_path, "state")

    make_preview(tmp_path, settings=settings)
    plan = make_preview(tmp_path, settings=settings)

    assert [operation.source_relative for operation in plan.operations] == ["notes.txt"]


def test_media_filename_normalization_preserves_title_year_and_series_identity() -> None:
    movie = normalize_media_filename(Path("Da_Vincis_Demons_2023_1080p_BluRay.mkv"))
    episode = normalize_media_filename(Path("Da_Vincis_Demons_S01E01_720p_BluRay_2023.mkv"))

    assert movie == "Da Vincis Demons 2023"
    assert episode == "Da Vincis Demons TV 2023"
    assert movie != episode


def test_media_filename_normalization_groups_episodes_by_series() -> None:
    first = normalize_media_filename(Path("Da_Vincis_Demons_S01E01_720p_BluRay_2023.mkv"))
    second = normalize_media_filename(Path("Da_Vincis_Demons_S02E03_720p_BluRay_2023.mkv"))

    assert first == second


def test_media_filename_normalization_drops_release_tail_metadata() -> None:
    assert normalize_media_filename(Path("DC.2026.1080p.WEB-DL.Hindi.5.1.mkv")) == "DC 2026"
    assert normalize_media_filename(Path("Love.Of.Your.Life.2026.1080p.WEB-DL.Hindi.5.1.mkv")) == "Love Of Your Life 2026"


def test_explicit_operation_selection_wins_over_yes() -> None:
    plan = Plan(
        plan_id="plan-test",
        root_path="/files",
        mode="fuzzy",
        operations=[
            PlanOperation(
                operation_id="operation-one",
                file_id="file-one",
                source_relative="one.txt",
                destination_relative="_organized/one.txt",
                content_hash="sha256:one",
                category="Documents",
                subject="One",
                confidence=0.8,
                reason="test",
            ),
            PlanOperation(
                operation_id="operation-two",
                file_id="file-two",
                source_relative="two.txt",
                destination_relative="_organized/two.txt",
                content_hash="sha256:two",
                category="Documents",
                subject="Two",
                confidence=0.8,
                reason="test",
            ),
        ],
    )

    assert _select_operations(plan, ["operation-one"], yes=True) == {"operation-one"}


def test_repeated_preview_keeps_operation_ids_stable(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("motion and force", encoding="utf-8")
    settings = settings_for(tmp_path)

    first = make_preview(tmp_path, settings=settings)
    second = make_preview(tmp_path, settings=settings)

    assert first.plan_id == second.plan_id
    assert [item.operation_id for item in first.operations] == [item.operation_id for item in second.operations]


def test_move_record_failure_rolls_back_the_file_move(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("motion and force", encoding="utf-8")
    plan = make_preview(tmp_path, settings=settings_for(tmp_path))
    operation = plan.operations[0]

    monkeypatch.setattr(operations, "_record_move", lambda record, storage_dir: (_ for _ in ()).throw(OperationError("storage failure")))

    with pytest.raises(OperationError, match="rolled back"):
        apply_plan(plan, tmp_path / ".smartfile", {operation.operation_id}, approved=True)

    assert source.is_file()
    assert not (tmp_path / operation.destination_relative).exists()


def test_plan_artifact_rejects_traversal_id(tmp_path: Path) -> None:
    plan = Plan(plan_id="plan-test", root_path="/files", mode="fuzzy")

    with pytest.raises(OperationError, match="Invalid artifact ID"):
        save_plan(plan.model_copy(update={"plan_id": "../outside"}), tmp_path / ".smartfile")
