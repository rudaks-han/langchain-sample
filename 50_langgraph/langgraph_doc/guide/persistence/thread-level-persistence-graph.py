from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

from langgraph.graph import StateGraph, MessagesState, START


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile()

# input_message = {"type": "user", "content": "안녕, 나는 홍길동이야"}
# for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
#     chunk["messages"][-1].pretty_print()
#
# input_message = {"type": "user", "content": "내 이름이 뭐야?"}
# for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
#     chunk["messages"][-1].pretty_print()

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
# LangGraph Cloud나 LangGraph Studio를 사용한다면, 자동으로 설정되므로 컴파일 할 때 checkpointer를 전달할 필요가 없다.

config = {"configurable": {"thread_id": "1"}}
input_message = {"type": "user", "content": "안녕, 나는 홍길동이야"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

input_message = {"type": "user", "content": "내 이름이 뭐야?"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

input_message = {"type": "user", "content": "내 이름이 뭐야?"}
for chunk in graph.stream(
    {"messages": [input_message]},
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
