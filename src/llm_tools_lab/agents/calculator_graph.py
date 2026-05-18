from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from llm_tools_lab.tools.calculator import AVAILABLE_FUNCTIONS, MATH_TOOLS


# Local interfaces
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls_count: int
    cancelled: bool


# Initializations
llm = ChatOllama(model="qwen3:8b")
llm = llm.bind_tools(MATH_TOOLS)


def call_llm(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def call_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    messages_tool_response = []
    for tc in last_message.tool_calls:
        # Get function result
        result = AVAILABLE_FUNCTIONS[tc["name"]](**tc["args"])
        # Add to messages
        messages_tool_response.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )
    return {
        "messages": messages_tool_response,
        "tool_calls_count": state["tool_calls_count"] + len(last_message.tool_calls),
    }


def format_response(state: AgentState) -> dict:
    last_content = state["messages"][-1].content
    messages = [
        SystemMessage(
            content="Respond in plain text only. No markdown, no bold, no headers, no LaTeX."
        ),
        HumanMessage(
            content=f"Reformat this response as plain text:\n\n{last_content}"
        ),
    ]
    response = llm.invoke(messages)
    return {"messages": [response]}


def human_review(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    print("\n[HUMAN REVIEW] El modelo quiere ejecutar estas tools:")
    for tc in last_message.tool_calls:
        print(f"  - {tc['name']}({tc['args']})\n")
    # User confirm
    confirmation = input("\n¿Continuar? (yes/no): ").strip().lower()
    if confirmation == "yes":
        return {"cancelled": False}
    else:
        return {
            "messages": [AIMessage(content="Operation cancelled by user.")],
            "cancelled": True,
        }


# Edges functions
# ------------------------------
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "human_review"
    return "format_response"


def after_review(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message["cancelled"]:
        return "format_response"
    return "tools"


# Build workflow
workflow = StateGraph(AgentState)
# Add nodes
workflow.add_node("llm", call_llm)
workflow.add_node("tools", call_tools)
workflow.add_node("format_response", format_response)
workflow.add_node("human_review", human_review)
# Add edges
workflow.add_edge(START, "llm")
workflow.add_conditional_edges("llm", should_continue)
workflow.add_conditional_edges("human_review", after_review)
workflow.add_edge("tools", "llm")
workflow.add_edge("format_response", END)
# Build workflow
graph = workflow.compile()
