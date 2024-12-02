from typing import Literal

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

load_dotenv()


class WeatherResponse(BaseModel):
    """Respond to the user with this"""

    temperature: float = Field(description="The temperature in celsius")
    wind_directon: str = Field(
        description="The direction of the wind in abbreviated form"
    )
    wind_speed: float = Field(description="The speed of the wind in km/h")


class AgentState(MessagesState):
    final_response: WeatherResponse


@tool
def get_weather(city: Literal["서울", "부산"]):
    """Use this to get weather information."""
    if city == "서울":
        return "서울은 구름이 많고 5 mph 북동풍이 불고 있고 기온은 30도이다."
    elif city == "부산":
        return "부산은 5 mph 남동풍이 불고 있고 32도이고 맑다."
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

model = ChatOpenAI(model="gpt-4o-mini")

model_with_tools = model.bind_tools(tools)
model_with_structured_output = model.with_structured_output(WeatherResponse)

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage


def call_model(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def respond(state: AgentState):
    response = model_with_structured_output.invoke(
        [HumanMessage(content=state["messages"][-2].content)]
    )
    return {"final_response": response}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "respond"
    else:
        return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("respond", respond)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "respond": "respond",
    },
)

workflow.add_edge("tools", "agent")
workflow.add_edge("respond", END)
graph = workflow.compile()

answer = graph.invoke(input={"messages": [("human", "서울 날씨 어때?")]})[
    "final_response"
]

print(answer)
