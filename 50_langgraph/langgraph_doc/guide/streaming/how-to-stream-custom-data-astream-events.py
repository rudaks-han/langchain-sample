import asyncio

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph


async def my_node(state: MessagesState, config: RunnableConfig):
    chunks = [
        "Four",
        "score",
        "and",
        "seven",
        "years",
        "ago",
        "our",
        "fathers",
        "...",
    ]
    for chunk in chunks:
        await adispatch_custom_event(
            "my_custom_event",
            {"chunk": chunk},
            config=config,  # <-- propagate config
        )

    return {"messages": [AIMessage(content=" ".join(chunks))]}


workflow = StateGraph(MessagesState)

workflow.add_node("model", my_node)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

app = workflow.compile()

from langchain_core.messages import HumanMessage

inputs = [HumanMessage(content="무슨 생각 하고 있어?")]


async def stream_content():
    async for event in app.astream_events({"messages": inputs}, version="v2"):
        tags = event.get("tags", [])
        if event["event"] == "on_custom_event" and event["name"] == "my_custom_event":
            data = event["data"]
            if data:
                print(data["chunk"], end="|", flush=True)


asyncio.run(stream_content())
