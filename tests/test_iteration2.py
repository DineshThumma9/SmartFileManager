from pathlib import Path

import ollama
from rich.console import Console

from smartfileman.embeddings.ollama import OllamaEmbeddingProvider
from smartfileman.readers import extract_units
from smartfileman.rendering import render_plan
from smartfileman.schemas import Plan, PlanOperation


def test_long_text_is_chunked_without_truncation(tmp_path: Path) -> None:
    source = tmp_path / "long.txt"
    text = " ".join(f"topic-{index}" for index in range(1000))
    source.write_text(text, encoding="utf-8")

    units = extract_units(source, max_chunk_chars=120)

    assert len(units) > 1
    assert "topic-999" in " ".join(unit.text for unit in units)
    assert all(len(unit.text) <= 120 for unit in units)
    assert all("chunk_index" in unit.locator for unit in units)


def test_ollama_embed_accepts_embed_response_object() -> None:
    provider = OllamaEmbeddingProvider("http://localhost:11434", "test-model")

    class FakeClient:
        def embed(self, **kwargs):
            assert kwargs["truncate"] is False
            return ollama.EmbedResponse(embeddings=[[1.0, 0.0], [0.0, 1.0]])

    provider.client = FakeClient()

    assert provider.embed(["one", "two"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert provider.dimension == 2


def test_visual_preview_groups_operations() -> None:
    plan = Plan(
        plan_id="plan-test",
        root_path="/files",
        mode="semantic",
        operations=[
            PlanOperation(
                operation_id="operation-1",
                file_id="file-1",
                source_relative="lecture.pdf",
                destination_relative="Force - Motion/Documents/lecture.pdf",
                content_hash="sha256:one",
                category="Documents",
                subject="Force - Motion",
                confidence=0.9,
                reason="Cluster evidence: force motion",
            )
        ],
    )
    console = Console(record=True, width=120)

    render_plan(console, plan)

    output = console.export_text()
    assert "SMARTFILE PREVIEW" in output
    assert "Force - Motion" in output
    assert "lecture.pdf" in output
    assert "90%" in output
