from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."


@tool
def get_coolest_cities():
    """Get a list of coolest cities"""
    return "nyc, sf"


tools = [get_weather, get_coolest_cities]
tool_node = ToolNode(tools)

model_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

result = model_with_tools.invoke("what's the weather in sf?").tool_calls
print(result)

result = tool_node.invoke(
    {"messages": [model_with_tools.invoke("what's the weather in sf?")]}
)
print(result)
