from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode


@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location in ["서울", "인천"]:
        return "현재 기온은 20도이고 구름이 많아."
    else:
        return "현재 기온은 30도이며 맑아"


@tool
def get_coolest_cities():
    """Get a list of coolest cities"""
    return "서울, 인천"


tools = [get_weather, get_coolest_cities]
tool_node = ToolNode(tools)

message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"location": "서울"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ],
)

result = tool_node.invoke({"messages": [message_with_single_tool_call]})
print(result)

message_with_multiple_tool_calls = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_coolest_cities",
            "args": {},
            "id": "tool_call_id_1",
            "type": "tool_call",
        },
        {
            "name": "get_weather",
            "args": {"location": "서울"},
            "id": "tool_call_id_2",
            "type": "tool_call",
        },
    ],
)

result = tool_node.invoke({"messages": [message_with_multiple_tool_calls]})
print(result)
