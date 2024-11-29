from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()


@tool
def search(query: str):
    """Call to surf the web."""
    return ["서울 날씨 좋아요~ 😈."]


tools = [search]
tool_node = ToolNode(tools)


model = ChatOpenAI(model="gpt-4o-mini")
model = model.bind_tools(tools)


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")

memory = MemorySaver()

app = workflow.compile(checkpointer=memory, interrupt_before=["action"])

from langchain_core.messages import HumanMessage

thread = {"configurable": {"thread_id": "3"}}
inputs = [HumanMessage(content="지금 서울 날씨 알려줘")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

current_state = app.get_state(thread)
last_message = current_state.values["messages"][-1]
last_message.tool_calls[0]["args"] = {"query": "현재 서울 날씨"}

result = app.update_state(thread, {"messages": last_message})
print(result)

current_state = app.get_state(thread).values["messages"][-1].tool_calls
print(current_state)

for event in app.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
