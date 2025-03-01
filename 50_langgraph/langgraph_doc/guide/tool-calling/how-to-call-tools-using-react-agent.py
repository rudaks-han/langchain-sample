from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
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


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

app = workflow.compile()

from IPython.display import Image, display

try:
    display(
        Image(
            app.get_graph().draw_mermaid_png(
                output_file_path="how-to-call-tools-using-toolnode-with-chatmodel.png"
            )
        )
    )
except Exception:
    pass

for chunk in app.stream(
    {"messages": [("human", "서울 날씨 어때?")]}, stream_mode="values"
):
    chunk["messages"][-1].pretty_print()


for chunk in app.stream(
    {"messages": [("human", "가장 추운 도시 날씨 어때?")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
