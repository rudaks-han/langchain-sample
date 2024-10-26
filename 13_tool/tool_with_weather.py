from typing import Tuple

import langchain
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import PydanticToolsParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI

langchain.debug = True

load_dotenv()


@tool
def get_weather(location: str) -> str:
    """Get weather condition."""
    # 날씨 api 호출
    result = f"{location}의 날씨는 맑음입니다."
    return result


llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)
# ai_msg = llm_with_tools.invoke("서울의 날씨는?")
# print(ai_msg)

query = "서울의 날씨는?"

messages = [("system", "You are a helpful assistant."), ("human", query)]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool = {"get_weather": get_weather}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

print(f"messages: {messages}")
