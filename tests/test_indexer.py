from pathlib import Path

from sqlmodel import select

from smartfileman.db import create_database, open_session
from smartfileman.embeddings import FakeEmbeddingProvider
from smartfileman.indexer import index_scan
from smartfileman.indexer import _embedding_payload
from smartfileman.models import EmbeddingRecord
from smartfileman.scanner import scan
from smartfileman.schemas import ContentUnit, ScannedFile


def test_indexer_reuses_unchanged_unit_embeddings(tmp_path: Path) -> None:
    source = tmp_path / "physics.txt"
    source.write_text("motion and force", encoding="utf-8")
    database = tmp_path / ".smartfile" / "index.sqlite3"
    provider = FakeEmbeddingProvider()

    first = index_scan(scan(tmp_path), database, provider)
    second = index_scan(scan(tmp_path), database, provider)

    assert first["embedded"] == 1
    assert second["embedded"] == 0
    assert second["reused"] == 1

    source.write_text("motion and anime", encoding="utf-8")
    third = index_scan(scan(tmp_path), database, provider)
    assert third["embedded"] == 1

    engine = create_database(database)
    with open_session(engine) as session:
        # Two body embeddings (the original and the edited text) plus the
        # identity embedding for each version of the file.
        assert len(session.exec(select(EmbeddingRecord)).all()) == 4


def test_indexer_stores_a_separate_identity_vector_per_file(tmp_path: Path) -> None:
    source = tmp_path / "physics.txt"
    source.write_text("motion and force", encoding="utf-8")
    database = tmp_path / ".smartfile" / "index.sqlite3"
    provider = FakeEmbeddingProvider()

    first = index_scan(scan(tmp_path), database, provider)
    second = index_scan(scan(tmp_path), database, provider)

    assert first["identity_embedded"] == 1
    assert first["embedded"] == 1
    assert second["identity_embedded"] == 0
    assert second["identity_reused"] == 1
    assert second["embedded"] == 0


def test_indexer_reuses_chunks_when_a_new_paragraph_shifts_ordinals(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("alpha topic\n\nbeta topic\n\ngamma topic", encoding="utf-8")
    database = tmp_path / ".smartfile" / "index.sqlite3"
    provider = FakeEmbeddingProvider()

    first = index_scan(scan(tmp_path), database, provider)
    source.write_text("new topic\n\nalpha topic\n\nbeta topic\n\ngamma topic", encoding="utf-8")
    second = index_scan(scan(tmp_path), database, provider)

    assert first["embedded"] == 3
    assert second["embedded"] == 1
    assert second["reused"] == 3


def test_indexer_batches_missing_units_across_files(tmp_path: Path) -> None:
    for index in range(20):
        (tmp_path / f"note-{index}.txt").write_text(f"topic {index}", encoding="utf-8")

    class RecordingProvider(FakeEmbeddingProvider):
        def __init__(self):
            super().__init__()
            self.batch_sizes: list[int] = []

        def embed(self, texts):
            self.batch_sizes.append(len(texts))
            return super().embed(texts)

    provider = RecordingProvider()

    result = index_scan(scan(tmp_path), tmp_path / ".smartfile" / "index.sqlite3", provider)

    assert result["embedded"] == 20
    # The body chunks are batched first, then the identity vectors are batched
    # on their own, so both pass through the provider in bounded groups.
    assert provider.batch_sizes == [16, 4, 16, 4]
    assert result["identity_embedded"] == 20


def test_indexer_does_not_reuse_embedding_from_a_different_model_digest(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("motion and force", encoding="utf-8")
    database = tmp_path / ".smartfile" / "index.sqlite3"

    first_provider = FakeEmbeddingProvider()
    index_scan(scan(tmp_path), database, first_provider)

    second_provider = FakeEmbeddingProvider()
    second_provider.model_digest = "fake-v2"
    result = index_scan(scan(tmp_path), database, second_provider)

    assert result["embedded"] == 1
    assert result["reused"] == 0


def test_semantic_embedding_payload_ignores_source_directory_names() -> None:
    file = ScannedFile(
        file_id="file",
        root_path="/files",
        relative_path="2027 - Com - Github/DineshThumma.pdf",
        filename="DineshThumma.pdf",
        extension=".pdf",
        mime_type="application/pdf",
        content_hash="sha256:file",
        size_bytes=1,
        modified_time_ns=1,
        units=[],
    )
    unit = ContentUnit(
        unit_type="page",
        ordinal=0,
        content_hash="sha256:unit",
        text="Professional summary and software engineering experience",
    )

    payload = _embedding_payload(file, unit)

    assert "DineshThumma.pdf" in payload
    assert "2027 - Com - Github" not in payload
