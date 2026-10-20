from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    model_digest: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...
