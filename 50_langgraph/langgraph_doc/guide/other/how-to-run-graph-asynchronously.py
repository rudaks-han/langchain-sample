import asyncio
from typing import Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


from langchain_core.tools import tool


@tool
def search(query: str):
    """Call to surf the web."""
    return ["질문에 대한 답은 내부에 있어"]


tools = [search]

from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo-1106")

model = model.bind_tools(tools)

from typing import Literal


def should_continue(state: State) -> Literal["end", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


async def call_model(state: State):
    messages = state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response]}


from langgraph.graph import END, StateGraph, START

workflow = StateGraph(State)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")

app = workflow.compile()

from IPython.display import Image, display

display(
    Image(
        app.get_graph().draw_mermaid_png(
            output_file_path="how-to-run-graph-asynchronously.png"
        )
    )
)

from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="서울 날씨 어때?")]}


async def run():
    result = await app.ainvoke(inputs)
    print(result)


# asyncio.run(run())

inputs = {"messages": [HumanMessage(content="서울 날씨 어때?")]}


async def run():
    async for output in app.astream(inputs, stream_mode="updates"):
        # stream_mode="updates"는 노드 이름으로 키가 지정된 dict 출력을 생성한다.
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value["messages"][-1].pretty_print())
        print("\n---\n")


# asyncio.run(run())


async def run():
    inputs = {"messages": [HumanMessage(content="서울 날씨 어때?")]}
    async for output in app.astream_log(inputs, include_types=["llm"]):
        # astream_log()은 요청된 로그 (여기서는 LLM)를 JSONPatch 형식으로 생성한다.
        for op in output.ops:
            if op["path"] == "/streamed_output/-":
                # this is the output from .stream()
                ...
            elif op["path"].startswith("/logs/") and op["path"].endswith(
                "/streamed_output/-"
            ):
                try:
                    content = op["value"].content[0]
                    if "partial_json" in content:
                        print(content["partial_json"], end="|")
                    elif "text" in content:
                        print(content["text"], end="|")
                    else:
                        print(content, end="|")
                except:
                    pass


asyncio.run(run())
