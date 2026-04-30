from llm_tools_lab.models.calculator import CalculatorResponse
from llm_tools_lab.services.ollama_client import run_agent
from llm_tools_lab.tools.calculator import MATH_TOOLS
import instructor
from instructor.exceptions import InstructorRetryException


def get_structured_calculation(
    user_request: str, model: str = "qwen3:8b"
) -> CalculatorResponse:
    # Run agent operation
    op_result = run_agent(user_message=user_request, model=model, tools=MATH_TOOLS)
    # Creates instructor client
    client = instructor.from_provider(f"ollama/{model}", async_client=False)
    try:
        # Calls to model
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a math assistant. Extract the calculation steps and final result from the provided text.",
                },
                {
                    "role": "user",
                    "content": f"{op_result}",
                },
            ],
            response_model=CalculatorResponse,
            max_retries=3,
        )
        return response
    except InstructorRetryException as e:
        raise ValueError(
            f"Model failed to return valid structured output after retries: {e}"
        ) from e
