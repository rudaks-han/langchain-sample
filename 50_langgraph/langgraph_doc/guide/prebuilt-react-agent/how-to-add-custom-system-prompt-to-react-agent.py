from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


from typing import Literal

from langchain_core.tools import tool


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

# 여기에 시스템 프롬프트를 추가한다

prompt = "영어로 대답해줘"

from langgraph.prebuilt import create_react_agent

graph = create_react_agent(model, tools=tools, state_modifier=prompt)


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "서울 날씨 어때?")]}

print_stream(graph.stream(inputs, stream_mode="values"))
