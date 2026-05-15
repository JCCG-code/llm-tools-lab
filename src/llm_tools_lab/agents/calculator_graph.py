from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from llm_tools_lab.tools.calculator import AVAILABLE_FUNCTIONS, MATH_TOOLS


# Local interfaces
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


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
    return {"messages": messages_tool_response}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# Build workflow
workflow = StateGraph(AgentState)
# Add nodes
workflow.add_node("llm", call_llm)
workflow.add_node("tools", call_tools)
# Add edges
workflow.add_edge(START, "llm")
workflow.add_conditional_edges("llm", should_continue)
workflow.add_edge("tools", "llm")
# Build workflow
graph = workflow.compile()
