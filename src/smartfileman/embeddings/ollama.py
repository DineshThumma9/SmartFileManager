from collections.abc import Sequence
from numbers import Real

import ollama


class OllamaEmbeddingProvider:
    provider_name = "ollama"

    def __init__(self, host: str, model_name: str):
        self.client = ollama.Client(host=host)
        self.model_name = model_name
        self.model_digest = "unknown"
        self._digest_resolved = False
        self.dimension = 0

    def resolve_model_digest(self) -> str:
        if self._digest_resolved:
            return self.model_digest
        self._digest_resolved = True
        try:
            response = self.client.list()
            models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
            for model in models:
                if isinstance(model, dict):
                    name = model.get("model") or model.get("name")
                    digest = model.get("digest")
                else:
                    name = getattr(model, "model", None) or getattr(model, "name", None)
                    digest = getattr(model, "digest", None)
                if name == self.model_name and digest:
                    self.model_digest = str(digest)
                    break
        except Exception:
            # An unresolved digest is deliberately conservative: the indexer
            # will not reuse unknown persistent model outputs on a later run.
            self.model_digest = "unknown"
        return self.model_digest

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.client.embed(
                model=self.model_name,
                input=list(texts),
                truncate=False,
            )
            if isinstance(response, dict):
                vectors = response.get("embeddings")
            else:
                vectors = getattr(response, "embeddings", None)
        except Exception as error:
            raise RuntimeError(f"Ollama embedding failed for {self.model_name}: {error}") from error

        if vectors is None or not isinstance(vectors, Sequence) or len(vectors) != len(texts):
            raise RuntimeError(
                f"Ollama returned {len(vectors) if isinstance(vectors, Sequence) else 'no'} "
                f"embeddings for {len(texts)} inputs."
            )
        vectors = list(vectors)
        if vectors and any(
            not isinstance(vector, (list, tuple))
            or not vector
            or any(not isinstance(value, Real) for value in vector)
            for vector in vectors
        ):
            raise RuntimeError("Ollama returned a malformed embedding vector.")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            raise RuntimeError("Ollama returned vectors with inconsistent dimensions.")
        self.dimension = len(vectors[0]) if vectors else self.dimension
        return [list(vector) for vector in vectors]
