from collections.abc import AsyncGenerator, Callable, Iterator

from ollama import ChatResponse, Message, chat

from llm_tools_lab.tools.registry import get_tools_for_message

# Const
RECURSION_LIMIT = 10


def run_agent(
    user_message: str,
    model: str = "qwen3:8b",
    tools: list[Callable] | None = None,
    think: bool = False,
) -> str:
    # Create message array
    messages = [{"role": "user", "content": user_message}]
    # Extract tools for message
    if tools is None:
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
            think=think,
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


async def stream_agent(
    user_message: str,
    model: str = "qwen3:8b",
    tools: list[Callable] | None = None,
    think: bool = False,
) -> AsyncGenerator[str, None]:
    # Create message array
    messages: list[dict] = [{"role": "user", "content": user_message}]
    # Extract tools for message
    if tools is None:
        tools = get_tools_for_message(user_message)
        # Checks tools
        if not tools:
            raise ValueError(f"No tools found for message: {user_message}")
    # Construct dict of tools
    available = {fn.__name__: fn for fn in tools}
    # Agent loop indefinite
    for _ in range(RECURSION_LIMIT):
        # Chat call streaming mode
        stream: Iterator[ChatResponse] = chat(
            model=model, messages=messages, tools=tools, think=think, stream=True
        )
        thinking = ""
        content = ""
        tool_calls: list[Message.ToolCall] = []
        # Acc partial fields
        for chunk in stream:
            if chunk.message.thinking:
                thinking += chunk.message.thinking
            if chunk.message.content:
                content += chunk.message.content
                yield chunk.message.content
            if chunk.message.tool_calls:
                tool_calls.extend(chunk.message.tool_calls)
        # Append accumulated fields to the messages
        if thinking or content or tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "thinking": thinking,
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )
        # Check tool calls
        if tool_calls:
            for tc in tool_calls:
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
