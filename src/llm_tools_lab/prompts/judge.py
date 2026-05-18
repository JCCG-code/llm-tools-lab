import instructor
from instructor import Mode
from instructor.core import InstructorRetryException

from llm_tools_lab.models.judge import JudgeResponse

JUDGE_PROMPT = """You are an expert evaluator. Score the following response.

Question: {question}
Context: {context}
Response: {response}

Evaluate on these criteria:
- Accuracy (1-5): Is the response factually correct based on the context?
- Relevance (1-5): Does the response answer the question?
- Completeness (1-5): Is the response complete?

Respond in this exact format:
accuracy: <score>
relevance: <score>
completeness: <score>
reasoning: <brief explanation>
"""


def judge_response(
    question: str,
    context: str,
    response: str,
    model: str = "qwen3:8b",
) -> JudgeResponse:
    """Evaluate a RAG response using LLM-as-judge pattern."""
    # Creates instructor client
    client = instructor.from_provider(
        f"ollama/{model}", async_client=False, mode=Mode.JSON
    )
    try:
        prompt = JUDGE_PROMPT.format(
            question=question, context=context, response=response
        )
        # Calls to model
        res: JudgeResponse = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            response_model=JudgeResponse,
            max_retries=3,
        )
        return res
    except InstructorRetryException as e:
        raise ValueError(
            f"Model failed to return valid structured output after retries: {e}"
        ) from e
