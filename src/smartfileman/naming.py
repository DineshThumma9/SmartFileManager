import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import ollama

from .readers import MEDIA_EXTENSIONS, normalize_media_filename


_MAX_EVIDENCE_CHARS = 2_000
_MAX_GENERATED_TOKENS = 32
_DEFAULT_FALLBACK = "Needs review"
_NAMING_CACHE_VERSION = "naming-v6-corrective-purpose-naming"
_NAME_TOKEN_RE = re.compile(r"[a-z0-9]+")


class ClusterNamer(Protocol):
    def name(self, evidence: str, fallback: str) -> str:
        ...


def safe_cluster_name(value: str, fallback: str) -> str:
    cleaned = _sanitize_name(value)
    if cleaned:
        return cleaned
    return _sanitize_name(fallback) or _DEFAULT_FALLBACK


def _sanitize_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = re.sub(r"[\\/:*?\"<>|\x00-\x1f\x7f]+", " ", value).strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:80].strip(" .")


def _compact_evidence(value: object) -> str:
    if not isinstance(value, str):
        return ""
    lines = [" ".join(line.split()) for line in value.splitlines()]
    compact = "\n".join(line for line in lines if line)
    return compact[:_MAX_EVIDENCE_CHARS].rstrip()


def _tokens(value: str) -> set[str]:
    return set(_NAME_TOKEN_RE.findall(value.lower()))


def _shared_evidence_tokens(evidence: str) -> set[str]:
    shared: set[str] = set()
    for line in evidence.splitlines():
        if line.lower().startswith(("shared topic/tool terms:", "shared identity/title terms:")):
            shared.update(_tokens(line.split(":", 1)[1]))
    return shared


def _name_is_supported(name: str, evidence: str) -> bool:
    candidate = _tokens(name)
    supported = _shared_evidence_tokens(evidence)
    if not candidate or not supported:
        return True
    overlap = candidate.intersection(supported)
    return bool(overlap) and len(overlap) / len(candidate) >= 0.5


class DeterministicClusterNamer:
    def name(self, evidence: str, fallback: str) -> str:
        return safe_cluster_name(fallback, "Needs review")


class OllamaClusterNamer:
    """Best-effort Gemma naming after membership has already been decided."""

    def __init__(
        self,
        host: str,
        model_name: str | None = None,
        cache_path: Path | None = None,
        timeout_seconds: float | None = 30.0,
    ):
        self.client = ollama.Client(host=host, timeout=timeout_seconds)
        self.model_name = model_name or ""
        self._resolved_model: str | None = None
        self.cache_path = cache_path
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, str]:
        if self.cache_path is None or not self.cache_path.is_file():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
            if payload.get("version") != _NAMING_CACHE_VERSION or not isinstance(entries, dict):
                return {}
            return {
                str(key): value
                for key, value in entries.items()
                if isinstance(value, str) and value.strip()
            }
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            return {}

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(f"{self.cache_path.suffix}.tmp")
        payload = {"version": _NAMING_CACHE_VERSION, "entries": self._cache}
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(self.cache_path)

    def _cache_key(self, model_name: str, evidence: str, fallback: str) -> str:
        canonical = "\n".join(
            (_NAMING_CACHE_VERSION, model_name, str(_MAX_GENERATED_TOKENS), evidence, fallback)
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _resolve_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        if self.model_name:
            self._resolved_model = self.model_name
            return self._resolved_model
        response = self.client.list()
        models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
        names: list[str] = []
        for model in models:
            if isinstance(model, dict):
                name = model.get("name") or model.get("model")
            else:
                name = getattr(model, "model", None) or getattr(model, "name", None)
            if name:
                names.append(str(name))
        candidates = [name for name in names if "embedding" not in name.lower() and "embed" not in name.lower()]
        gemma = [name for name in candidates if "gemma" in name.lower()]
        if gemma:
            self._resolved_model = gemma[0]
            return self._resolved_model
        if candidates:
            self._resolved_model = candidates[0]
            return self._resolved_model
        raise RuntimeError("No local Ollama naming model is installed.")

    def _chat_name(self, model_name: str, evidence: str, fallback: str, correction: bool = False) -> str:
        correction_hint = (
            " The previous answer was unsupported. Return a new label using at least one or two exact terms "
            "from the shared evidence lines; do not use filenames, source folders, dates, URLs, usernames, "
            "or release metadata."
            if correction
            else ""
        )
        response = self.client.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Name a file organization cluster. Return JSON only with a short 'name' value. "
                        "Name the shared document purpose, subject, or identity—not the storage location. "
                        "Use repeated evidence first. Ignore source directories, storage headings, dates, years, "
                        "URLs, email addresses, usernames, platform names, and media release metadata. "
                        "Do not copy a unique filename or one-file heading. Prefer a natural two-to-four-word "
                        "purpose label over a list of tools. Do not invent details or add a category. "
                        "The JSON object must use exactly this shape: {\"name\": \"short label\"}."
                        + correction_hint
                    ),
                },
                {"role": "user", "content": f"Evidence:\n{evidence}\nFallback: {fallback}"},
            ],
            think=False,
            format={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            options={"temperature": 0, "num_predict": _MAX_GENERATED_TOKENS},
        )
        message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Ollama naming response must be a JSON object")
        name = parsed.get("name") or parsed.get("cluster")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Ollama naming response must contain a non-empty string name")
        return name

    def name(self, evidence: str, fallback: str) -> str:
        safe_fallback = safe_cluster_name(fallback, _DEFAULT_FALLBACK)
        try:
            model_name = self._resolve_model()
            compact_evidence = _compact_evidence(evidence)
            cache_key = self._cache_key(model_name, compact_evidence, safe_fallback)
            cached = self._cache.get(cache_key)
            if cached:
                return safe_cluster_name(cached, safe_fallback)
            name = self._chat_name(model_name, compact_evidence, safe_fallback)
            if not _name_is_supported(name, compact_evidence):
                name = self._chat_name(model_name, compact_evidence, safe_fallback, correction=True)
            if not _name_is_supported(name, compact_evidence):
                return safe_fallback
            result = safe_cluster_name(name, safe_fallback)
            self._cache[cache_key] = result
            try:
                self._save_cache()
            except OSError:
                pass
            return result
        except Exception:
            return safe_fallback


def representative_evidence(
    files: Sequence[object],
    topic_terms: Sequence[str],
    identity_terms: Sequence[str] = (),
) -> str:
    excerpts: list[str] = []
    for file in files[:8]:
        relative_path = getattr(file, "relative_path", "")
        filename = Path(str(relative_path)).name or str(getattr(file, "filename", relative_path))
        extension = str(getattr(file, "extension", "")).lower()
        if extension in MEDIA_EXTENSIONS:
            filename = normalize_media_filename(Path(getattr(file, "filename", relative_path)))
            filename_label = "Filename/title evidence"
        else:
            filename_label = "Filename/path"
        units = getattr(file, "units", [])
        texts = [str(getattr(unit, "text", "")) for unit in units if getattr(unit, "text", "")]
        opening_lines = [line.strip() for line in texts[0].splitlines() if line.strip()] if texts else []
        opening = " ".join(opening_lines[:2])[:240]
        normalized_text = " ".join(" ".join(text.split()) for text in texts)
        excerpt = normalized_text[:320]
        if opening and excerpt.startswith(opening):
            excerpt = excerpt[len(opening):].strip()
        excerpts.append(
            f"{filename_label}: {filename}\n"
            f"Identity/opening evidence: {opening}\n"
            f"Body/topic evidence: {excerpt}"
        )
    return _compact_evidence(
        f"Shared topic/tool terms: {', '.join(topic_terms)}\n"
        f"Shared identity/title terms: {', '.join(identity_terms)}\n"
        + "\n".join(excerpts)
    )
