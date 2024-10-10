from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class InputState(TypedDict):
    user_input: str


class OutputState(TypedDict):
    graph_output: str


class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str


class PrivateState(TypedDict):
    bar: str


def node_1(state: InputState) -> OverallState:
    return {"foo": state["user_input"] + " name"}


def node_2(state: OverallState) -> PrivateState:
    return {"bar": state["foo"] + " is"}


def node_3(state: PrivateState) -> OutputState:
    return {"graph_output": state["bar"] + " Lance"}


builder = StateGraph(OverallState, input=InputState, output=OutputState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()
result = graph.invoke({"user_input": "My"})
print(result)  # {"graph_output": "My name is Lance"}
