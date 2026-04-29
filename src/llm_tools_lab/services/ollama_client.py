from ollama import chat, ChatResponse
from llm_tools_lab.tools.registry import get_tools_for_message

# Const
RECURSION_LIMIT = 10


def run_agent(user_message: str, model: str = "qwen3:8b") -> str:
    # Create message array
    messages = [{"role": "user", "content": user_message}]
    # Extract tools for message
    tools = get_tools_for_message(user_message)
    # Checks tools
    if not tools:
        raise ValueError(f"No tools found for message: {user_message}")
    # Construct dict of tools
    available = {fn.__name__: fn for fn in tools}
    # Agent loop indefinite
    for _ in range(RECURSION_LIMIT):
        # First thinking iteration
        response: ChatResponse = chat(
            model=model,
            messages=messages,
            tools=tools,
            think=True,
        )
        messages.append(response.message.model_dump(exclude_none=True))
        # Check tool calls
        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                # Extract data
                function_name = tc.function.name
                function_args = tc.function.arguments
                # Checks functions detected
                if function_name in available:
                    result = available[function_name](**function_args)
                    # Add tool to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": function_name,
                            "content": str(result),
                        }
                    )
        else:
            break
    else:
        raise RecursionError(f"Agent exceed maximum iterations ({RECURSION_LIMIT})")
    # Return check
    if not response.message.content:
        raise ValueError("Model returned empty response")
    return response.message.content
