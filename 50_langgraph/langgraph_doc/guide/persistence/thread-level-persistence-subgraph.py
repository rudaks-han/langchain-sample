from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph


# 서브 그래프
class SubgraphState(TypedDict):
    foo: str  # 이 키는 부모 그래프 상태와 공유되어 있다.
    bar: str


def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}


def subgraph_node_2(state: SubgraphState):
    # 이 노드는 서브그래프에서만 사용 가능한 상태 키 ('bar')를 사용하고 있으며, 공유 상태 키 ('foo')에 대한 업데이트를 보내고 있다.
    return {"foo": state["foo"] + state["bar"]}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()


# 부모 그래프
class State(TypedDict):
    foo: str


def node_1(state: State):
    return {"foo": "hi! " + state["foo"]}


builder = StateGraph(State)
builder.add_node("node_1", node_1)
# 서브그래프를 부모 그래프에 노드로 추가하는 것을 볼 수 있다.
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")

checkpointer = MemorySaver()
# 부모 그래프를 컴파일할 때만 checkpointer를 전달해야 한다.
# LangGraph는 자동으로 checkpointer를 하위 서브그래프로 전파한다.
graph = builder.compile(checkpointer=checkpointer)

from IPython.display import Image, display

display(
    Image(
        graph.get_graph().draw_mermaid_png(
            output_file_path="./thread-level-persistence-subgraph.png"
        )
    )
)

config = {"configurable": {"thread_id": "1"}}

for _, chunk in graph.stream({"foo": "foo"}, config, subgraphs=True):
    print(chunk)

print(graph.get_state(config).values)

state_with_subgraph = [
    s for s in graph.get_state_history(config) if s.next == ("node_2",)
][0]

subgraph_config = state_with_subgraph.tasks[0].state
print(subgraph_config)

print(graph.get_state(subgraph_config).values)
