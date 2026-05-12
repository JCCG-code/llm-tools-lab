# RAG Module
# ─────────────────────────────────────────────────────
# INDEXING:   file_indexer → indexer → [chunker + embeddings] → Qdrant
# QUERYING:   rag_agent → retriever → Qdrant → LLM → response
# INFRA:      vector_store (Qdrant connection + collection setup)
