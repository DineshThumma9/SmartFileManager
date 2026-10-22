from dataclasses import dataclass
from pathlib import Path

import pytest

from smartfileman.naming import DeterministicClusterNamer, OllamaClusterNamer, representative_evidence, safe_cluster_name


def test_safe_cluster_name_removes_unsafe_characters_and_bounds_length() -> None:
    value = "  Project:/Notes?\x00 " + ("x" * 100)

    result = safe_cluster_name(value, "fallback")

    assert result == "Project Notes " + ("x" * 66)
    assert len(result) == 80


def test_deterministic_namer_sanitizes_fallback() -> None:
    assert DeterministicClusterNamer().name("ignored", "Needs:/review") == "Needs review"


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        '{"name": "   "}',
        '{"name": 42}',
        "[]",
        "not json",
    ],
)
def test_invalid_ai_name_uses_deterministic_fallback(content: str) -> None:
    namer = OllamaClusterNamer("http://localhost:11434", "gemma-test")

    class FakeClient:
        def chat(self, **kwargs):
            return {"message": {"content": content}}

    namer.client = FakeClient()

    assert namer.name("Terms: project", "project-notes") == "project-notes"


def test_ai_naming_bounds_generation_and_compacts_evidence() -> None:
    namer = OllamaClusterNamer("http://localhost:11434", "gemma-test")
    captured: dict[str, object] = {}

    class FakeClient:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return {"message": {"content": '{"name": "Project Notes"}'}}

    namer.client = FakeClient()
    evidence = "Terms: project\n" + ("  project evidence   " * 500)

    assert namer.name(evidence, "project-notes") == "Project Notes"

    assert captured["options"] == {"temperature": 0, "num_predict": 32}
    assert captured["think"] is False
    # Structured output is deliberately not requested. Ollama's JSON-schema
    # constrained decoding is roughly an order of magnitude slower than a
    # plain completion, and the slower path regularly exceeded the request
    # timeout, so every naming attempt fell back to the deterministic label.
    assert "format" not in captured
    messages = captured["messages"]
    assert isinstance(messages, list)
    user_content = messages[1]["content"]
    assert len(user_content) <= 2_050
    assert "  " not in user_content


def test_ai_naming_accepts_attribute_style_ollama_response() -> None:
    namer = OllamaClusterNamer("http://localhost:11434", "gemma-test")

    @dataclass
    class Message:
        content: str

    @dataclass
    class Response:
        message: Message

    class FakeClient:
        def chat(self, **kwargs):
            return Response(Message('{"name": "Resume Materials"}'))

    namer.client = FakeClient()

    assert namer.name("Terms: resume", "resume") == "Resume Materials"


def test_ai_naming_accepts_cluster_key_from_ollama() -> None:
    namer = OllamaClusterNamer("http://localhost:11434", "gemma-test")

    class FakeClient:
        def chat(self, **kwargs):
            return {"message": {"content": '{"cluster": "Resume Materials"}'}}

    namer.client = FakeClient()

    assert namer.name("Terms: resume", "resume") == "Resume Materials"


def test_ai_naming_retries_an_unsupported_label() -> None:
    namer = OllamaClusterNamer("http://localhost:11434", "gemma-test")
    calls: list[dict[str, object]] = []

    class FakeClient:
        def chat(self, **kwargs):
            calls.append(kwargs)
            name = "2027 - Com - Github" if len(calls) == 1 else "Backend Engineering"
            return {"message": {"content": f'{{"name": "{name}"}}'}}

    namer.client = FakeClient()
    evidence = "Shared topic/tool terms: java, backend, boot, databases\nShared identity/title terms: professional, summary"

    assert namer.name(evidence, "Java - Backend - Databases") == "Backend Engineering"
    assert len(calls) == 2
    assert "previous answer was unsupported" in calls[1]["messages"][0]["content"]


def test_ai_naming_reuses_successful_persistent_name_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / ".smartfile" / "naming-cache.json"
    first = OllamaClusterNamer("http://localhost:11434", "gemma-test", cache_path=cache_path)
    calls = 0

    class FirstClient:
        def chat(self, **kwargs):
            nonlocal calls
            calls += 1
            return {"message": {"content": '{"name": "Resume Materials"}'}}

    first.client = FirstClient()
    assert first.name("Terms: resume", "resume") == "Resume Materials"

    second = OllamaClusterNamer("http://localhost:11434", "gemma-test", cache_path=cache_path)

    class CachedClient:
        def chat(self, **kwargs):
            raise AssertionError("cached names must not call Ollama")

    second.client = CachedClient()
    assert second.name("Terms: resume", "resume") == "Resume Materials"
    assert calls == 1


def test_representative_evidence_separates_filename_opening_and_body() -> None:
    class Unit:
        text = "Dinesh\nSoftware Engineer\nReact and Neo4j"

    class File:
        relative_path = "Dinesh_Resume.pdf"
        units = [Unit()]

    evidence = representative_evidence([File()], ["react", "neo4j"], ["software", "engineer"])

    assert "Filename/path: Dinesh_Resume.pdf" in evidence
    assert "Shared topic/tool terms: react, neo4j" in evidence
    assert "Shared identity/title terms: software, engineer" in evidence
    assert "Identity/opening evidence: Dinesh Software Engineer" in evidence
    assert "Body/topic evidence: React and Neo4j" in evidence


def test_representative_evidence_does_not_feed_source_directories_to_namer() -> None:
    class Unit:
        text = "Professional Summary\nBackend engineer\nReact and Neo4j"

    class File:
        relative_path = "2027 - Com - Github/DineshThumma.pdf"
        units = [Unit()]

    evidence = representative_evidence([File()], ["react", "neo4j"], ["professional", "summary"])

    assert "Filename/path: DineshThumma.pdf" in evidence
    assert "2027 - Com - Github" not in evidence


def test_representative_evidence_normalizes_media_titles_before_naming() -> None:
    class Unit:
        text = "DC 2026"

    class File:
        filename = "DC.2026.1080p.WEB-DL.Hindi.5.1.mkv"
        relative_path = "Telegram Desktop/DC.2026.1080p.WEB-DL.Hindi.5.1.mkv"
        extension = ".mkv"
        units = [Unit()]

    evidence = representative_evidence([File()], ["dc", "2026"], ["dc", "2026"])

    assert "Filename/title evidence: DC 2026" in evidence
    assert "Hindi" not in evidence
