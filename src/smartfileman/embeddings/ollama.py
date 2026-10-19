from typing import Sequence

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
            response = self.client.embed(model=self.model_name, input=list(texts))
            vectors = response.get("embeddings") if isinstance(response, dict) else None
        except Exception as error:
            raise RuntimeError(f"Ollama embedding failed for {self.model_name}: {error}") from error

        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("Ollama returned an unexpected embedding response.")
        if vectors and not isinstance(vectors[0], list):
            raise RuntimeError("Ollama returned a malformed embedding vector.")
        self.dimension = len(vectors[0]) if vectors else self.dimension
        return vectors
