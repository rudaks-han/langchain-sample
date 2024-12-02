from typing import Literal

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from typing_extensions import TypedDict

load_dotenv()


@tool
def get_weather(city: str):
    """Get the weather for a specific city"""
    return f"{city}은 맑아요"


raw_model = ChatOpenAI()
model = raw_model.with_structured_output(get_weather)


class SubGraphState(MessagesState):
    city: str


def model_node(state: SubGraphState):
    result = model.invoke(state["messages"])
    return {"city": result["city"]}


def weather_node(state: SubGraphState):
    result = get_weather.invoke({"city": state["city"]})
    return {"messages": [{"role": "assistant", "content": result}]}


subgraph = StateGraph(SubGraphState)
subgraph.add_node(model_node)
subgraph.add_node(weather_node)
subgraph.add_edge(START, "model_node")
subgraph.add_edge("model_node", "weather_node")
subgraph.add_edge("weather_node", END)
subgraph = subgraph.compile(interrupt_before=["weather_node"])

memory = MemorySaver()


class RouterState(MessagesState):
    route: Literal["weather", "other"]


class Router(TypedDict):
    route: Literal["weather", "other"]


router_model = raw_model.with_structured_output(Router)


def router_node(state: RouterState):
    system_message = "다음 질문이 날씨에 관한 것인지 아닌지 분류해줘."
    messages = [{"role": "system", "content": system_message}] + state["messages"]
    route = router_model.invoke(messages)
    return {"route": route["route"]}


def normal_llm_node(state: RouterState):
    response = raw_model.invoke(state["messages"])
    return {"messages": [response]}


def route_after_prediction(
    state: RouterState,
) -> Literal["weather_graph", "normal_llm_node"]:
    if state["route"] == "weather":
        return "weather_graph"
    else:
        return "normal_llm_node"


graph = StateGraph(RouterState)
graph.add_node(router_node)
graph.add_node(normal_llm_node)
graph.add_node("weather_graph", subgraph)
graph.add_edge(START, "router_node")
graph.add_conditional_edges("router_node", route_after_prediction)
graph.add_edge("normal_llm_node", END)
graph.add_edge("weather_graph", END)
graph = graph.compile()

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()


class GrandfatherState(MessagesState):
    to_continue: bool


def router_node(state: GrandfatherState):
    return {"to_continue": True}


def route_after_prediction(state: GrandfatherState):
    if state["to_continue"]:
        return "graph"
    else:
        return END


grandparent_graph = StateGraph(GrandfatherState)
grandparent_graph.add_node(router_node)
grandparent_graph.add_node("graph", graph)
grandparent_graph.add_edge(START, "router_node")
grandparent_graph.add_conditional_edges(
    "router_node", route_after_prediction, ["graph", END]
)
grandparent_graph.add_edge("graph", END)
grandparent_graph = grandparent_graph.compile(checkpointer=MemorySaver())

from IPython.display import Image, display

display(
    Image(
        grandparent_graph.get_graph(xray=2).draw_mermaid_png(
            output_file_path="how-to-view-and-update-state-in-subgraphs-double-nested-subgraph.png"
        )
    )
)

config = {"configurable": {"thread_id": "2"}}
inputs = {"messages": [{"role": "user", "content": "서울 날씨 어때?"}]}
for update in grandparent_graph.stream(
    inputs, config=config, stream_mode="updates", subgraphs=True
):
    print(update)

state = grandparent_graph.get_state(config, subgraphs=True)
print("Grandparent State:")
print(state.values)
print("---------------")
print("Parent Graph State:")
print(state.tasks[0].state.values)
print("---------------")
print("Subgraph State:")
print(state.tasks[0].state.tasks[0].state.values)

grandparent_graph_state = state
parent_graph_state = grandparent_graph_state.tasks[0].state
subgraph_state = parent_graph_state.tasks[0].state
grandparent_graph.update_state(
    subgraph_state.config,
    {"messages": [{"role": "assistant", "content": "비와요"}]},
    as_node="weather_node",
)
for update in grandparent_graph.stream(
    None, config=config, stream_mode="updates", subgraphs=True
):
    print(update)

print(grandparent_graph.get_state(config).values["messages"])

for state in grandparent_graph.get_state_history(config):
    print(state)
    print("-----")
