import hashlib
import math
from typing import Sequence


class FakeEmbeddingProvider:
    provider_name = "fake"
    model_name = "fake-v1"
    model_digest = "fake-v1"
    dimension = 16

    def __init__(self):
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += len(texts)
        vectors = []
        for text in texts:
            values = []
            for index in range(self.dimension):
                digest = hashlib.sha256(f"{index}:{text}".encode("utf-8")).digest()
                values.append((int.from_bytes(digest[:4], "big") / 2**32) * 2 - 1)
            length = math.sqrt(sum(value * value for value in values)) or 1
            vectors.append([value / length for value in values])
        return vectors
