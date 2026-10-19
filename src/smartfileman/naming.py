import json
import re
from collections.abc import Sequence
from typing import Protocol


class ClusterNamer(Protocol):
    def name(self, evidence: str, fallback: str) -> str:
        ...


def safe_cluster_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", " ", value).strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:80].strip(" .") or fallback


class DeterministicClusterNamer:
    def name(self, evidence: str, fallback: str) -> str:
        return safe_cluster_name(fallback, "Needs review")


class OllamaClusterNamer:
    """Best-effort Gemma naming after membership has already been decided."""

    def __init__(self, host: str, model_name: str | None = None):
        import ollama

        self.client = ollama.Client(host=host)
        self.model_name = model_name or ""
        self._resolved_model: str | None = None

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
            name = model.get("name") if isinstance(model, dict) else getattr(model, "model", None)
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

    def name(self, evidence: str, fallback: str) -> str:
        try:
            response = self.client.chat(
                model=self._resolve_model(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Name a file topic cluster. Return JSON only with a short 'name' value. "
                            "Use only the evidence. Do not invent details, use path separators, or add a category."
                        ),
                    },
                    {"role": "user", "content": f"Evidence:\n{evidence}\nFallback: {fallback}"},
                ],
                format="json",
                options={"temperature": 0},
            )
            message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
            parsed = json.loads(content)
            return safe_cluster_name(str(parsed.get("name", "")), fallback)
        except Exception:
            return safe_cluster_name(fallback, "Needs review")


def representative_evidence(files: Sequence[object], terms: Sequence[str]) -> str:
    excerpts: list[str] = []
    for file in files[:8]:
        relative_path = getattr(file, "relative_path", "")
        units = getattr(file, "units", [])
        excerpt = " ".join(getattr(unit, "text", "") for unit in units)[:400]
        excerpts.append(f"{relative_path}: {excerpt}")
    return f"Terms: {', '.join(terms)}\n" + "\n".join(excerpts)
