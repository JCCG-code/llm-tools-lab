from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from llm_tools_lab.rag.retriever import search
from llm_tools_lab.rag.vector_store import COLLECTION_NAME, get_client

# Reranker model
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # standard constant


# Initializations
_reranker = CrossEncoder(RERANKER_MODEL)


def _reciprocal_rank_fusion(
    semantic_results: list[dict], bm25_results: list[dict]
) -> list[dict]:
    """Combine two ranked lists using RRF algorithm."""
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    for rank, semantic_result in enumerate(semantic_results):
        text = semantic_result["text"]
        scores[text] = 1 / (RRF_K + rank)
        chunks[text] = semantic_result

    for rank, bm25_result in enumerate(bm25_results):
        text = bm25_result["text"]
        scores[text] = scores.get(text, 0) + 1 / (RRF_K + rank)
        chunks[text] = bm25_result

    res = []
    for text, rrf_score in scores.items():
        chunk_data = {**chunks[text], "rrf_score": rrf_score}
        res.append(chunk_data)

    res.sort(key=lambda x: x["rrf_score"], reverse=True)
    return res


def _bm25_search(query: str, chunks: list[str], top_k: int) -> list[dict]:
    """Search using BM25 keyword matching."""
    # Tokenizer query and chunks
    tokenized_query = query.split()
    tokenized_chunks = [chunk.split() for chunk in chunks]
    # bm25 instance with tokenized chunks
    bm25 = BM25Okapi(tokenized_chunks)
    # Query scores
    query_scores = bm25.get_scores(tokenized_query)
    # Get top chunks
    chunk_scored = []
    for chunk, score in zip(chunks, query_scores):
        if float(score) > 0:
            chunk_scored.append({"text": chunk, "score": float(score)})
    # Sort chunk scored array
    chunk_scored.sort(key=lambda x: x["score"], reverse=True)
    return chunk_scored[:top_k]


def hybrid_search(
    query: str,
    top_k: int = 5,
    rerank: bool = True,
) -> list[dict]:
    """
    Hybrid search combining semantic + BM25 + optional reranking.
    """
    # Get all chunks
    client = get_client()
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    all_chunks = [
        point.payload["text"] for point in points if point.payload is not None
    ]
    # Semantic search in Qdrant
    semantic_search = search(query, top_k=top_k * 2)
    # BM25 search
    bm25_search = _bm25_search(query, all_chunks, top_k=top_k * 2)
    # Combines both techniques
    combined_search = _reciprocal_rank_fusion(semantic_search, bm25_search)
    # Check rerank
    if rerank:
        # Creates vector of (query, chunk) pairs
        pairs = [[query, chunk["text"]] for chunk in combined_search]
        # Predict scores from pairs
        scores = _reranker.predict(pairs)
        # Add rerank score to res
        for chunk, score in zip(combined_search, scores):
            chunk["rerank_score"] = float(score)
        # Sort vector
        combined_search.sort(key=lambda x: x["rerank_score"], reverse=True)
    else:
        combined_search.sort(key=lambda x: x["rrf_score"], reverse=True)
    return combined_search[:top_k]
