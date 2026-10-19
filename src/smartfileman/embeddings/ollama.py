from collections.abc import Sequence
from numbers import Real

import ollama


class OllamaEmbeddingProvider:
    provider_name = "ollama"

    def __init__(self, host: str, model_name: str):
        self.client = ollama.Client(host=host)
        self.model_name = model_name
        self.dimension = 0

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
