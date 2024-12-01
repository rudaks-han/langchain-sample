import asyncio

from langchain_core.messages import AIMessage
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.types import StreamWriter


async def my_node(
    state: MessagesState,
    writer: StreamWriter,  # <-- chunk가 스트리밍되도록 StreamWriter를 제공한다.
):
    chunks = [
        "87년",
        "전",
        ",",
        "우리",
        "의",
        "선조",
        "들",
        "께서",
        "...",
    ]
    for chunk in chunks:
        # stream_mode=custom을 사용하여 스트리밍될 chunk를 작성한다.
        writer(chunk)

    return {"messages": [AIMessage(content=" ".join(chunks))]}


workflow = StateGraph(MessagesState)

workflow.add_node("model", my_node)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

app = workflow.compile()

from langchain_core.messages import HumanMessage

inputs = [HumanMessage(content="무슨 생각 하고 있어?")]


async def stream_content():
    async for chunk in app.astream(
        {"messages": inputs}, stream_mode=["custom", "updates"]
    ):
        print(chunk, flush=True)


asyncio.run(stream_content())
