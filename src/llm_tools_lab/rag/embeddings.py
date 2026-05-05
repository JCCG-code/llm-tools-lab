from collections.abc import Sequence

from ollama import embeddings


def embed(text: str, model: str = "nomic-embed-text") -> Sequence[float]:
    result = embeddings(model, prompt=text)
    return result.embedding
