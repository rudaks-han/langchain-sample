from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START, MessagesState

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

from typing import Literal
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver


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
graph = graph.compile(checkpointer=memory)

from IPython.display import Image, display

display(
    Image(
        graph.get_graph(xray=1).draw_mermaid_png(
            output_file_path="how-to-view-and-update-state-in-subgraphs.png"
        )
    )
)

config = {"configurable": {"thread_id": "1"}}
inputs = {"messages": [{"role": "user", "content": "안녕!"}]}
# for update in graph.stream(inputs, config=config, stream_mode="updates"):
#     print(update)

config = {"configurable": {"thread_id": "2"}}
inputs = {"messages": [{"role": "user", "content": "서울 날씨 어때?"}]}
# for update in graph.stream(inputs, config=config, stream_mode="updates"):
#     print(update)

config = {"configurable": {"thread_id": "3"}}
inputs = {"messages": [{"role": "user", "content": "서울 날씨 어때?"}]}
for update in graph.stream(inputs, config=config, stream_mode="values", subgraphs=True):
    print(update)

state = graph.get_state(config)
print(state.next)

print(state.tasks)

state = graph.get_state(config, subgraphs=True)
print(state.tasks[0])

for update in graph.stream(None, config=config, stream_mode="values", subgraphs=True):
    print(update)
#
parent_graph_state_before_subgraph = next(
    h for h in graph.get_state_history(config) if h.next == ("weather_graph",)
)

subgraph_state_before_model_node = next(
    h
    for h in graph.get_state_history(parent_graph_state_before_subgraph.tasks[0].state)
    if h.next == ("model_node",)
)

# 이 패턴은 얼마나 깊이 들어가도 확장될 수 있다.
# subsubgraph_stat_history = next(h for h in graph.get_state_history(subgraph_state_before_model_node.tasks[0].state) if h.next == ('my_subsubgraph_node',))

print(subgraph_state_before_model_node.next)

for value in graph.stream(
    None,
    config=subgraph_state_before_model_node.config,
    stream_mode="values",
    subgraphs=True,
):
    print(value)


print("---- 4 ----")
config = {"configurable": {"thread_id": "4"}}
inputs = {"messages": [{"role": "user", "content": "서울 날씨 어때?"}]}
for update in graph.stream(inputs, config=config, stream_mode="updates"):
    print(update)

state = graph.get_state(config, subgraphs=True)
print(state.values["messages"])

graph.update_state(state.tasks[0].state.config, {"city": "부산"})

for update in graph.stream(None, config=config, stream_mode="updates", subgraphs=True):
    print(update)

print("----- 14 -----")
config = {"configurable": {"thread_id": "14"}}
inputs = {"messages": [{"role": "user", "content": "서울 날씨 어때?"}]}
for update in graph.stream(
    inputs, config=config, stream_mode="updates", subgraphs=True
):
    print(update)
# Graph execution should stop before the weather node
print("interrupted!")

state = graph.get_state(config, subgraphs=True)

# We update the state by passing in the message we want returned from the weather node, and make sure to use as_node
graph.update_state(
    state.tasks[0].state.config,
    {"messages": [{"role": "assistant", "content": "비와요"}]},
    as_node="weather_node",
)
for update in graph.stream(None, config=config, stream_mode="updates", subgraphs=True):
    print(update)

print(graph.get_state(config).values["messages"])

print("----- 8 -----")
config = {"configurable": {"thread_id": "8"}}
inputs = {"messages": [{"role": "user", "content": "서울 날씨 어때?"}]}
for update in graph.stream(
    inputs, config=config, stream_mode="updates", subgraphs=True
):
    print(update)
# Graph execution should stop before the weather node
print("interrupted!")

# We update the state by passing in the message we want returned from the weather graph, making sure to use as_node
# Note that we don't need to pass in the subgraph config, since we aren't updating the state inside the subgraph
graph.update_state(
    config,
    {"messages": [{"role": "assistant", "content": "rainy"}]},
    as_node="weather_graph",
)
for update in graph.stream(None, config=config, stream_mode="updates"):
    print(update)

print(graph.get_state(config).values["messages"])
