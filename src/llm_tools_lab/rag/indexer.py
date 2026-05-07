from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from llm_tools_lab.rag.chunker import chunk_text
from llm_tools_lab.rag.embeddings import embed
from llm_tools_lab.rag.vector_store import (
    COLLECTION_NAME,
    create_collection,
    get_client,
)


def index_text(text: str, source: str = "unknown") -> int:
    """Chunk, embed and index text into Qdrant. Returns number of chunks indexed."""
    # Creates client and collection
    client = get_client()
    create_collection(client)
    points: list[PointStruct] = []
    # Generates chunks
    chunks_text = chunk_text(text)
    for i, chunk in enumerate(chunks_text):
        # Generates embedding
        embedded_chunk = list(embed(chunk))
        # Point struct to qdrant
        points.append(
            PointStruct(
                id=i, vector=embedded_chunk, payload={"text": chunk, "source": source}
            )
        )
    # Insert points
    client.upsert(COLLECTION_NAME, points=points)
    # Return numer of indexed points
    return len(points)
