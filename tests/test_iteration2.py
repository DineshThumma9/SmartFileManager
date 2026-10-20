from pathlib import Path

import ollama
from rich.console import Console

from smartfileman.embeddings.ollama import OllamaEmbeddingProvider
from smartfileman.naming import OllamaClusterNamer
from smartfileman.planner import _semantic_labels
from smartfileman.readers import extract_units
from smartfileman.rendering import render_plan
from smartfileman.schemas import ContentUnit, Plan, PlanOperation, ScannedFile


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
                reason="Cluster similarity evidence: force, motion; peer-supported cluster",
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
    assert "Evidence: force motion" in output
    assert "cluster" not in output.lower().split("evidence:", 1)[1].split("\n", 1)[0]


def test_semantic_clusters_do_not_chain_unrelated_files() -> None:
    def file(file_id: str, name: str) -> ScannedFile:
        return ScannedFile(
            file_id=file_id,
            root_path="/files",
            relative_path=name,
            filename=name,
            extension=".txt",
            mime_type="text/plain",
            content_hash=f"sha256:{file_id}",
            size_bytes=1,
            modified_time_ns=1,
            units=[ContentUnit(unit_type="paragraph", ordinal=0, content_hash=f"sha256:{file_id}", text=name)],
        )

    files = [file("a", "alpha-notes.txt"), file("b", "alpha-related.txt"), file("c", "completely-different.txt")]
    labels = _semantic_labels(
        files,
        {"a": [1.0, 0.0], "b": [0.99, 0.1], "c": [0.0, 1.0]},
    )

    assert labels["a"][0] == labels["b"][0]
    assert labels["c"][0] != labels["a"][0]


def test_ollama_cluster_namer_uses_model_json() -> None:
    namer = OllamaClusterNamer("http://localhost:11434", "gemma-test")

    class FakeClient:
        def chat(self, **kwargs):
            return {"message": {"content": '{"name": "Resume Materials"}'}}

    namer.client = FakeClient()

    assert namer.name("Terms: resume, experience", "resume-experience") == "Resume Materials"
