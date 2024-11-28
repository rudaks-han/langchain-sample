import operator
from typing import Annotated

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    aggregate: Annotated[list, operator.add]


def node_a(state):
    print("I'm A")
    return {"aggregate": ["I'm A"]}


def node_b(state):
    print("I'm B")
    return {"aggregate": ["I'm B"]}


def node_c(state):
    print("I'm C")
    return {"aggregate": ["I'm C"]}


def node_d(state):
    print("I'm D")
    return {"aggregate": ["I'm A"]}


builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_edge(START, "a")
builder.add_node("b", node_b)
builder.add_node("c", node_c)
builder.add_node("d", node_d)
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()

from IPython.display import Image, display

display(
    Image(
        graph.get_graph().draw_mermaid_png(
            output_file_path="./graph_recursion_limit.png"
        )
    )
)

from langgraph.errors import GraphRecursionError

try:
    result = graph.invoke({"aggregate": []}, {"recursion_limit": 3})
    print(result)
except GraphRecursionError:
    print("Recursion Error")

# try:
#     graph.invoke({"aggregate": []}, {"recursion_limit": 4})
# except GraphRecursionError:
#     print("Recursion Error")
