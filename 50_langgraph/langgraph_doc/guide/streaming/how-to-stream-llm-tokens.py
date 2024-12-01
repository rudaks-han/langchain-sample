import asyncio
from typing import Annotated

from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# Add messages essentially does this with more
# robust handling
# def add_messages(left: list, right: list):
#     return left + right


class State(TypedDict):
    messages: Annotated[list, add_messages]


from langchain_core.tools import tool


@tool
def search(query: str):
    """Call to surf the web."""
    return ["흐려요~"]


tools = [search]

from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

model = model.bind_tools(tools)

from langchain_core.runnables import RunnableConfig

from langgraph.graph import END, START, StateGraph


def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "tools"


async def call_model(state: State, config: RunnableConfig):
    messages = state["messages"]
    response = await model.ainvoke(messages, config)
    return {"messages": response}


workflow = StateGraph(State)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    ["tools", END],
)

workflow.add_edge("tools", "agent")

app = workflow.compile()

from IPython.display import Image, display

display(
    Image(
        app.get_graph().draw_mermaid_png(
            output_file_path="how-to-stream-llm-tokens.png"
        )
    )
)

from langchain_core.messages import AIMessageChunk, HumanMessage

inputs = [HumanMessage(content="서울 날씨 어때?")]


async def stream_async():
    first = True
    async for msg, metadata in app.astream(
        {"messages": inputs}, stream_mode="messages"
    ):
        if msg.content and not isinstance(msg, HumanMessage):
            print(msg.content, end="|", flush=True)

        if isinstance(msg, AIMessageChunk):
            if first:
                gathered = msg
                first = False
            else:
                gathered = gathered + msg

            if msg.tool_call_chunks:
                print(gathered.tool_calls)


asyncio.run(stream_async())
