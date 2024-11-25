from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore

load_dotenv()

in_memory_store = InMemoryStore()

import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore


model = ChatOpenAI(model="gpt-4o-mini")


# 참고: 노드에 Store 매개변수를 전달한다.
# 이는 그래프를 컴파일할 때 사용하는 Store이다.
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)
    memories = store.search(namespace)
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    # 사용자가 모델에게 기억하라고 요청하면 새로운 기억을 저장한다.
    last_message = state["messages"][-1]
    if "기억해" in last_message.content.lower():
        memory = "이름은 홍길동"
        store.put(namespace, str(uuid.uuid4()), {"data": memory})

    response = model.invoke(
        [{"type": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

# 참고: 그래프를 컴파일할 때 Store 객체를 전달한다.
graph = builder.compile(checkpointer=MemorySaver(), store=in_memory_store)
# 만일 LangGraph Cloud나 LangGraph Studio를 사용하고 있다면, 그래프를 컴파일할 때 store나 checkpointer를 전달할 필요가 없다. 이는 자동으로 수행된다.

config = {"configurable": {"thread_id": "1", "user_id": "1"}}
input_message = {"type": "user", "content": "안녕. 기억해, 내 이름은 홍길동이야."}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "2", "user_id": "1"}}
input_message = {"type": "user", "content": "내 이름은 뭐야?"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

for memory in in_memory_store.search(("memories", "1")):
    print(memory.value)

config = {"configurable": {"thread_id": "3", "user_id": "2"}}
input_message = {"type": "user", "content": "내 이름은 뭐야?"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
