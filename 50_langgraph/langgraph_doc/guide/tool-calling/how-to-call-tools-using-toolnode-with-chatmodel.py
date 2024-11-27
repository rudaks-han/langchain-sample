from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

load_dotenv()


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

model_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

result = model_with_tools.invoke("서울 날씨 어때?").tool_calls
print(result)

result = tool_node.invoke({"messages": [model_with_tools.invoke("서울 날씨 어때?")]})
print(result)
