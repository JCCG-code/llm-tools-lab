from llm_tools_lab.rag.hybrid_retriever import hybrid_search
from llm_tools_lab.rag.retriever import search

queries = [
    "Who is the narrator of the story, and what is his relationship with the main characters?",
    "What does the green light that Gatsby looks at across the bay symbolize?",
    "How did Jay Gatsby acquire his immense fortune?",
    "In the climax of the novel, who was actually driving the car that hit and killed Myrtle Wilson, and who takes the blame?",
    "How does the novel critique the idea of the 'American Dream'?",
]


with open("BENCHMARK.md", "w", encoding="utf-8") as a:
    a.write("# The Great Gatsby - Benchmarks\n\n")

    for query in queries:
        a.write(f"## Query -> {query}\n\n")
        # Search
        a.write("### Normal search\n\n")
        for r in search(query=query, top_k=5):
            a.write(f"{r['score']:.3f} | {r['text']}\n\n")
        # Hybrid search
        a.write("### Hybrid search\n\n")
        for r in hybrid_search(query=query, top_k=5):
            a.write(f"{r['rerank_score']:.3f} | {r['text']}\n\n")
