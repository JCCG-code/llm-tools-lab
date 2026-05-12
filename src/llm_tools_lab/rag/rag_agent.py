from ollama import ChatResponse, chat

from llm_tools_lab.rag.hybrid_retriever import hybrid_search
from llm_tools_lab.rag.retriever import search


def text_rag_agent(text: str, model: str = "qwen3:8b", use_hybrid: bool = False) -> str:
    if use_hybrid:
        relevant_chunks = hybrid_search(text)
    else:
        relevant_chunks = search(text)
    prompt = "Context:\n"
    for chunk in relevant_chunks:
        prompt += chunk["text"] + "\n"
    # Add user prompt
    prompt += f"\n\nUser:\n{text}"
    print(prompt)
    # Calls to ollama model
    response: ChatResponse = chat(
        model,
        messages=[
            {
                "role": "system",
                "content": "Answer ONLY based on the provided context. If the"
                " information is not in the context, say 'I don't have that "
                "information'.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    if response.message.content is None:
        raise ValueError("Error in model response")
    return response.message.content
