import instructor
from instructor import Mode
from ollama import ChatResponse, chat

from llm_tools_lab.models.classification import ClassificationResponse

INFO = """
The player dribbles the ball very quickly -> Basketball
What a dunk that guy just made! -> Basketball

The player switches the ball to the other side with great precision -> Football
We need to move the ball around, guys -> Football

What a serve he hit! -> Tennis
The player just smashed his racket in frustration -> Tennis

Classify the following input with only the sport name, no explanation:
"""


def few_shot_prompt(
    user_input: str, info: str, model: str = "qwen3:8b"
) -> ClassificationResponse:
    client = instructor.from_provider(
        f"ollama/{model}", async_client=False, mode=Mode.JSON
    )
    res: ClassificationResponse = client.chat.completions.create(
        messages=[{"role": "user", "content": INFO + "\n\n" + user_input}],
        response_model=ClassificationResponse,
        max_retries=3,
    )
    return res
