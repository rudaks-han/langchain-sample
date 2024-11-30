from IPython.display import Image, display
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from typing_extensions import Literal

load_dotenv()


@tool
def weather_search(city: str):
    """Search for the weather"""
    print("----")
    print(f"{city}을 검색하고 있어요!")
    print("----")
    return "Sunny!"


model = ChatOpenAI(model_name="gpt-4o-mini").bind_tools([weather_search])


class State(MessagesState):
    """Simple state."""


def call_llm(state):
    return {"messages": [model.invoke(state["messages"])]}


def human_review_node(state):
    pass


def run_tool(state):
    new_messages = []
    tools = {"weather_search": weather_search}
    tool_calls = state["messages"][-1].tool_calls
    for tool_call in tool_calls:
        tool = tools[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        new_messages.append(
            {
                "role": "tool",
                "name": tool_call["name"],
                "content": result,
                "tool_call_id": tool_call["id"],
            }
        )
    return {"messages": new_messages}


def route_after_llm(state) -> Literal[END, "human_review_node"]:
    if len(state["messages"][-1].tool_calls) == 0:
        return END
    else:
        return "human_review_node"


def route_after_human(state) -> Literal["run_tool", "call_llm"]:
    if isinstance(state["messages"][-1], AIMessage):
        return "run_tool"
    else:
        return "call_llm"


builder = StateGraph(State)
builder.add_node(call_llm)
builder.add_node(run_tool)
builder.add_node(human_review_node)
builder.add_edge(START, "call_llm")
builder.add_conditional_edges("call_llm", route_after_llm)
builder.add_conditional_edges("human_review_node", route_after_human)
builder.add_edge("run_tool", "call_llm")

memory = MemorySaver()

graph = builder.compile(checkpointer=memory, interrupt_before=["human_review_node"])

display(
    Image(
        graph.get_graph().draw_mermaid_png(
            output_file_path="how-to-review-tool-calls.png"
        )
    )
)

initial_input = {"messages": [{"role": "user", "content": "서울 날씨 어때?"}]}

thread = {"configurable": {"thread_id": "2"}}

for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)

print("Pending Executions!")
print(graph.get_state(thread).next)

for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
