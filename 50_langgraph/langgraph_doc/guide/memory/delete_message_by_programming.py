from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import RemoveMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

memory = MemorySaver()


@tool
def search(query: str):
    """Call to surf the web."""
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."


tools = [search]
tool_node = ToolNode(tools)
model = ChatOpenAI(model_name="gpt-4o-mini")
bound_model = model.bind_tools(tools)


def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 3:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-3]]}


# 모델을 호출하는 함수를 정의한다.
def call_model(state: MessagesState):
    response = bound_model.invoke(state["messages"])
    # 리스트를 반환한다. 기존 리스트에 추가될 것이기 때문이다.
    return {"messages": response}


# 바로 끝내는 대신에 delete_messages를 호출하는 로직을 수정할 필요가 있다.
def should_continue(state: MessagesState) -> Literal["action", "delete_messages"]:
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    # 만일 함수 호출이 없다면, 끝낸다.
    if not last_message.tool_calls:
        return "delete_messages"
    # 만일 함수 호출이 있다면, 계속한다.
    return "action"


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# 이것이 우리가 정의하는 새로운 노드이다.
workflow.add_node(delete_messages)


workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
)
workflow.add_edge("action", "agent")

# 추가하고 있는 새로운 엣지이다. 메시지를 삭제한 후에 끝낸다.
workflow.add_edge("delete_messages", END)
app = workflow.compile(checkpointer=memory)

from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "3"}}
input_message = HumanMessage(content="안녕! 내 이름은 홍길동이야")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    print([(message.type, message.content) for message in event["messages"]])


input_message = HumanMessage(content="내 이름이 뭐야?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    print([(message.type, message.content) for message in event["messages"]])

messages = app.get_state(config).values["messages"]
print(messages)
