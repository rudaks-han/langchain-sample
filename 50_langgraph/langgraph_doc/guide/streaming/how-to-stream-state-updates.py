import asyncio
from typing import Literal

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()


@tool
def get_weather(city: Literal["서울", "부산"]):
    """Use this to get weather information."""
    if city == "서울":
        return "서울은 흐릴것 같아요"
    elif city == "부산":
        return "부산은 항상 맑아요"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
graph = create_react_agent(model, tools)

inputs = {"messages": [("human", "서울 날씨 어때?")]}


async def stream_async():
    async for chunk in graph.astream(inputs, stream_mode="updates"):
        for node, values in chunk.items():
            print(f"Receiving update from node: '{node}'")
            print(values)
            print("\n\n")


asyncio.run(stream_async())
