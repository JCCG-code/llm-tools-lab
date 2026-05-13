from langfuse import propagate_attributes
from ollama import ChatResponse, chat

from llm_tools_lab.observability.langfuse_client import get_langfuse
from llm_tools_lab.rag.hybrid_retriever import hybrid_search
from llm_tools_lab.rag.retriever import search


def text_rag_agent(
    text: str,
    model: str = "qwen3:8b",
    use_hybrid: bool = False,
    user_id: str | None = None,
    session_id: str | None = None,
) -> str:
    # Get langfuse
    langfuse = get_langfuse()
    # RAG Agent measures
    with langfuse.start_as_current_observation(
        as_type="span", name="rag_query"
    ) as root:
        root.update(
            input={"query": text, "model": model},
        )
        with propagate_attributes(user_id=user_id, session_id=session_id):
            # Retrieval
            with langfuse.start_as_current_observation(
                as_type="span", name="retrieval"
            ) as span:
                if use_hybrid:
                    relevant_chunks = hybrid_search(text)
                else:
                    relevant_chunks = search(text)
                span.update(
                    output={
                        "chunks_count": len(relevant_chunks),
                        "use_hybrid": use_hybrid,
                    }
                )
            # Prompt build
            with langfuse.start_as_current_observation(
                as_type="span", name="build_prompt"
            ) as span:
                prompt = "Context:\n"
                for chunk in relevant_chunks:
                    prompt += chunk["text"] + "\n"
                # Add user prompt
                prompt += f"\n\nUser:\n{text}"
                span.update(output={"prompt": prompt})
            # LLM Call
            with langfuse.start_as_current_observation(
                as_type="generation", name="llm_response", model=model
            ) as generation:
                generation.update(input={"prompt": prompt})
                # Calls to ollama model
                response: ChatResponse = chat(
                    model=model,
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
                generation.update(output={"llm_response": response.message.content})
    # Checks response
    if response.message.content is None:
        raise ValueError("Error in model response")
    # Close langfuse
    langfuse.flush()
    # Return statement
    return response.message.content
