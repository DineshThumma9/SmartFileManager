import hashlib
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import ollama

from .readers import MEDIA_EXTENSIONS, normalize_media_filename

logger = logging.getLogger(__name__)

_MAX_EVIDENCE_CHARS = 2_000
_MAX_GENERATED_TOKENS = 32
_DEFAULT_FALLBACK = "Needs review"
_NAMING_CACHE_VERSION = "naming-v7-purpose-naming"
_NAME_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")

# The naming model is asked for the shared document type and purpose. Local
# instruction-tuned models answer a "list the tools" prompt with a list of
# tools, so the prompt states the distinction explicitly and shows that the
# opening lines carry the document type.
_NAMING_SYSTEM_PROMPT = (
    "You label a group of related files so the label can be used as a folder name.\n"
    "Decide what these files ARE: their document type and purpose. A good label is a "
    "thing like a resume, a meeting note, a design document, a research paper, a tax "
    "return, a music album, or a project plan.\n"
    "The 'Shared topic/tool terms' line lists words that merely appear in the files. A "
    "programming language, a framework, a database, or a company name is usually NOT the "
    "right label, because many unrelated files share those words.\n"
    "The 'Identity/opening evidence' lines show how the files actually begin. That is the "
    "strongest signal for the document type, so use it first.\n"
    "Rules: name the shared type and purpose rather than a list of tools. Ignore source "
    "folder names, dates, years, URLs, usernames, and platform or release metadata. Do not "
    "copy one file's unique filename. Use one to four words. Do not add a category word.\n"
    'Answer with JSON only, exactly {"name": "your label"}.'
)


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
    """Tokens that several files in the cluster agree on.

    These are the only tokens a label is allowed to reuse verbatim. A purpose
    label such as "Resume" rarely appears in the shared topic terms, so the
    pool also carries shared identity terms and shared filename terms.
    """

    shared: set[str] = set()
    for line in evidence.splitlines():
        lowered = line.lower()
        if lowered.startswith(
            (
                "shared topic/tool terms:",
                "shared identity/title terms:",
                "shared filename terms:",
            )
        ):
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
            " The previous answer was unsupported. Return a new label that names what these "
            "files ARE (their document type and purpose), and reuse at least one or two exact "
            "terms from the shared evidence lines. Do not use filenames, source folders, dates, "
            "URLs, usernames, or release metadata."
            if correction
            else ""
        )
        # Structured output is deliberately not requested. Ollama's JSON-schema
        # constrained decoding is an order of magnitude slower than a plain
        # completion, and the slower path regularly exceeded the request
        # timeout, so every naming attempt failed and the deterministic
        # fallback was used instead.
        response = self.client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": _NAMING_SYSTEM_PROMPT + correction_hint},
                {"role": "user", "content": f"Evidence:\n{evidence}\nFallback: {fallback}"},
            ],
            think=False,
            options={"temperature": 0, "num_predict": _MAX_GENERATED_TOKENS},
        )
        message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        return _parse_name(content)

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
            supported = _name_is_supported(name, compact_evidence)
            if not supported:
                name = self._chat_name(model_name, compact_evidence, safe_fallback, correction=True)
                supported = _name_is_supported(name, compact_evidence)
            result = safe_cluster_name(name, safe_fallback)
            if supported:
                self._cache[cache_key] = result
                try:
                    self._save_cache()
                except OSError:
                    logger.warning("Could not persist the naming cache at %s", self.cache_path, exc_info=True)
            return result
        except Exception:
            # A failed naming call must stay visible. Silently returning the
            # fallback made it impossible to tell a model failure apart from a
            # model that had nothing to contribute.
            logger.warning(
                "Cluster naming failed; using the deterministic fallback instead.", exc_info=True
            )
            return safe_fallback


def _parse_name(content: object) -> str:
    """Read a name out of a model response that need not be strict JSON.

    The model is asked for JSON, but a local instruction-tuned model may wrap
    it in a code fence or add a short preamble. Those are tolerated; a
    response with no name in it is still an error, because silently treating
    arbitrary prose as a label would put text like "not json" into a folder
    name.
    """

    if not isinstance(content, str):
        raise ValueError("Ollama naming response must contain text")
    text = _FENCE_RE.sub("", content.strip()).strip()
    if not text:
        raise ValueError("Ollama naming response must contain text")
    payload: object = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None
    if isinstance(payload, dict):
        name = payload.get("name") or payload.get("cluster")
        if isinstance(name, str) and name.strip():
            return name
        raise ValueError("Ollama naming response must contain a non-empty string name")
    raise ValueError("Ollama naming response must contain a JSON object with a name")


def shared_filename_terms(files: Sequence[object], minimum_support: int = 2) -> list[str]:
    """Filename tokens that several files in a cluster agree on.

    These are supporting identity evidence, not a license to copy one file's
    unique name: a token must appear in at least ``minimum_support`` stems.
    """

    counts: dict[str, int] = {}
    for file in files:
        filename = str(getattr(file, "filename", ""))
        stem = Path(filename).stem or filename
        for token in _tokens(stem):
            counts[token] = counts.get(token, 0) + 1
    required = minimum_support if len(files) > 1 else 1
    ranked = sorted(
        (token for token, count in counts.items() if count >= required and len(token) > 2),
        key=lambda token: (-counts[token], token),
    )
    return ranked[:4]


def representative_evidence(
    files: Sequence[object],
    topic_terms: Sequence[str],
    identity_terms: Sequence[str] = (),
    filename_terms: Sequence[str] = (),
) -> str:
    excerpts: list[str] = []
    for file in files[:6]:
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
        f"Shared filename terms: {', '.join(filename_terms)}\n"
        + "\n".join(excerpts)
    )
