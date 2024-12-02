from langgraph.graph.state import StateGraph, START, END
from typing_extensions import TypedDict


class GrandChildState(TypedDict):
    my_grandchild_key: str


def grandchild_1(state: GrandChildState) -> GrandChildState:
    # NOTE: child or parent keys will not be accessible here
    return {"my_grandchild_key": state["my_grandchild_key"] + ", how are you"}


grandchild = StateGraph(GrandChildState)
grandchild.add_node("grandchild_1", grandchild_1)

grandchild.add_edge(START, "grandchild_1")
grandchild.add_edge("grandchild_1", END)

grandchild_graph = grandchild.compile()

result = grandchild_graph.invoke({"my_grandchild_key": "hi Bob"})
print(result)


class ChildState(TypedDict):
    my_child_key: str


def call_grandchild_graph(state: ChildState) -> ChildState:
    # state를 child state channels (`my_child_key`)에서 child state channels (`my_grandchild_key`)로 변환한다.
    grandchild_graph_input = {"my_grandchild_key": state["my_child_key"]}
    # 상태를 grandchild state channels (`my_grandchild_key`)에서 child state channels (`my_child_key`)로 변환한다.
    grandchild_graph_output = grandchild_graph.invoke(grandchild_graph_input)
    return {"my_child_key": grandchild_graph_output["my_grandchild_key"] + " today?"}


child = StateGraph(ChildState)
child.add_node("child_1", call_grandchild_graph)
child.add_edge(START, "child_1")
child.add_edge("child_1", END)
child_graph = child.compile()

result = child_graph.invoke({"my_child_key": "hi Bob"})
print(result)


class ParentState(TypedDict):
    my_key: str


def parent_1(state: ParentState) -> ParentState:
    # NOTE: child or grandchild keys won't be accessible here
    return {"my_key": "hi " + state["my_key"]}


def parent_2(state: ParentState) -> ParentState:
    return {"my_key": state["my_key"] + " bye!"}


def call_child_graph(state: ParentState) -> ParentState:
    # 상태를 parent state channels (`my_key`)에서 child state channels (`my_child_key`)로 변환한다.
    child_graph_input = {"my_child_key": state["my_key"]}
    # 상태를 child state channels (`my_child_key`)에서 parent state channels (`my_key`)로 변환한다.
    child_graph_output = child_graph.invoke(child_graph_input)
    return {"my_key": child_graph_output["my_child_key"]}


parent = StateGraph(ParentState)
parent.add_node("parent_1", parent_1)
parent.add_node("child", call_child_graph)
parent.add_node("parent_2", parent_2)

parent.add_edge(START, "parent_1")
parent.add_edge("parent_1", "child")
parent.add_edge("child", "parent_2")
parent.add_edge("parent_2", END)

parent_graph = parent.compile()

result = parent_graph.invoke({"my_key": "Bob"})
print(result)
