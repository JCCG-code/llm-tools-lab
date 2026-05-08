from llm_tools_lab.rag.embeddings import embed
from llm_tools_lab.rag.vector_store import (
    COLLECTION_NAME,
    get_client,
)


def search(query: str, top_k: int = 3, min_score: float = 0.5) -> list[dict]:
    """Search for similar chunks in Qdrant. Returns top_k most similar chunks."""
    # Creates client and Qdrant collection
    client = get_client()
    res: list[dict] = []
    # Embed query
    query_embedded = list(embed(query))
    # Search in db
    results = client.query_points(COLLECTION_NAME, query=query_embedded, limit=top_k)
    for point in results.points:
        # Payload is not none
        if point.payload is None:
            continue
        # Add to response data
        if point.score >= min_score:
            res.append(
                {
                    "text": point.payload["text"],
                    "source": point.payload["source"],
                    "score": point.score,
                }
            )
    # Return statement
    return res
