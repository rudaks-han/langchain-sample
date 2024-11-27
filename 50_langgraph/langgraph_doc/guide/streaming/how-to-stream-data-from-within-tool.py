import asyncio

from dotenv import load_dotenv
from langchain_core.callbacks import Callbacks
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()


@tool
async def get_items(
    place: str,
    callbacks: Callbacks,  # 직접 callbacks를 받는다 (Python <= 3.10인 경우는 필요하다)
) -> str:
    """Use this tool to look up which items are in the given place."""
    # async를 사용할 때는 LLM을 ainvoke를 사용하여 호출해야 한다!
    # 그렇지 않으면 스트리밍이 작동하지 않는다.
    return await llm.ainvoke(
        [
            {
                "role": "user",
                "content": f"Can you tell me what kind of items i might find in the following place: '{place}'. "
                "List at least 3 such items separating them by a comma. And include a brief description of each item..",
            }
        ],
        {"callbacks": callbacks},
    )


llm = ChatOpenAI(model_name="gpt-4o-mini")
tools = [get_items]
agent = create_react_agent(llm, tools=tools)


async def stream_content():
    final_message = ""
    async for msg, metadata in agent.astream(
        {"messages": [("human", "선반에 어떤 물건이 있어?")]},
        stream_mode="messages",
    ):
        # Stream all messages from the tool node
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and metadata["langgraph_node"] == "tools"
            and not msg.name
        ):
            print(msg.content, end="|", flush=True)
        # Final message should come from our agent
        if msg.content and metadata["langgraph_node"] == "agent":
            final_message += msg.content


# asyncio.run(stream_content())

from langchain_core.messages import HumanMessage


async def stream_content():
    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": "침실에 뭐가 있어?"}]}, version="v2"
    ):
        if (
            event["event"] == "on_chat_model_stream"
            and event["metadata"].get("langgraph_node") == "tools"
        ):
            print(event["data"]["chunk"].content, end="|", flush=True)


asyncio.run(stream_content())
