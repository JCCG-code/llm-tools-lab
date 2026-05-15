from llm_tools_lab.agents.calculator_graph import AgentState, graph

# Test 1 — operación simple
result = graph.invoke(
    AgentState(messages=["What is 15 plus 27?"], tool_calls_count=0, cancelled=False)
)
print("Test 1:", result["messages"][-1].content)
print("Tool calls:", result["tool_calls_count"])

# Test 2 — operación explícita encadenada
result = graph.invoke(
    AgentState(
        messages=["First multiply 345 by 345, then add 345 to the result"],
        tool_calls_count=0,
        cancelled=False,
    )
)
print("Test 2:", result["messages"][-1].content)
print("Tool calls:", result["tool_calls_count"])

# Test 3 — sin tools necesarias
result = graph.invoke(
    AgentState(
        messages=["What is the capital of France?"], tool_calls_count=0, cancelled=False
    )
)
print("Test 3:", result["messages"][-1].content)
print("Tool calls:", result["tool_calls_count"])
