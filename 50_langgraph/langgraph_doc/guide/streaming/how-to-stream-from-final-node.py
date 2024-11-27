import asyncio
from typing import Literal

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

load_dotenv()


@tool
def get_weather(city: Literal["서울", "부산"]):
    """Use this to get weather information."""
    if city == "서울":
        return "맑아요~"
    elif city == "부산":
        return "비와요~"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
final_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

model = model.bind_tools(tools)
# 마지막 노드에서 호출된 모델만 필터링하려면 모델 스트림 이벤트를 필터링하는 데 사용할 수 있는 태그를 추가하는 곳이다.
# 단일 LLM을 호출하는 경우 필요하지 않지만 노드 내에서 여러 모델을 호출하고 그 중 하나의 이벤트만 필터링하려는 경우 중요할 수 있다.
final_model = final_model.with_config(tags=["final_node"])
tool_node = ToolNode(tools=tools)

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage


def should_continue(state: MessagesState) -> Literal["tools", "final"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "final"


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def call_final_model(state: MessagesState):
    messages = state["messages"]
    last_ai_message = messages[-1]
    response = final_model.invoke(
        [
            SystemMessage("AI 로커 목소리로 다시 작성해주세요"),
            HumanMessage(last_ai_message.content),
        ]
    )
    response.id = last_ai_message.id
    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("final", call_final_model)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
)

builder.add_edge("tools", "agent")
builder.add_edge("final", END)

graph = builder.compile()

from IPython.display import display, Image

display(
    Image(
        graph.get_graph().draw_mermaid_png(
            output_file_path="how-to-stream-from-final-node.png"
        )
    )
)

from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="서울 날씨 어때?")]}
for msg, metadata in graph.stream(inputs, stream_mode="messages"):
    if (
        msg.content
        and not isinstance(msg, HumanMessage)
        and metadata["langgraph_node"] == "final"
    ):
        print(msg.content, end="|", flush=True)

print("_______")

inputs = {"messages": [HumanMessage(content="부산 날씨 어때?")]}


async def stream_content():
    async for event in graph.astream_events(inputs, version="v2"):
        kind = event["event"]
        tags = event.get("tags", [])
        # 커스텀 태그를 기반으로 필터링
        if kind == "on_chat_model_stream" and "final_node" in event.get("tags", []):
            data = event["data"]
            if data["chunk"].content:
                print(data["chunk"].content, end="|", flush=True)


asyncio.run(stream_content())
