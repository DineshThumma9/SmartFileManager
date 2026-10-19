from pathlib import Path

from sqlmodel import select

from smartfileman.db import create_database, open_session
from smartfileman.embeddings import FakeEmbeddingProvider
from smartfileman.indexer import index_scan
from smartfileman.models import EmbeddingRecord
from smartfileman.scanner import scan


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
        assert len(session.exec(select(EmbeddingRecord)).all()) == 2
