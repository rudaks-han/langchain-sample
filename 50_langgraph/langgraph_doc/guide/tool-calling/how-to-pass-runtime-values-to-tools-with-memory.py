from typing import Annotated, Tuple, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

load_dotenv()

doc_store = InMemoryStore()

namespace = ("documents", "1")  # user ID
doc_store.put(
    namespace, "doc_0", {"doc": "FooBar company just raised 1 Billion dollars!"}
)
namespace = ("documents", "2")  # user ID
doc_store.put(namespace, "doc_1", {"doc": "FooBar company was founded in 2019"})

from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedStore, ToolNode, create_react_agent


@tool
def get_context(
    question: str,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
) -> Tuple[str, List[Document]]:
    """Get relevant context for answering the question."""
    user_id = config.get("configurable", {}).get("user_id")
    docs = [item.value["doc"] for item in store.search(("documents", user_id))]
    return "\n\n".join(doc for doc in docs)


result = get_context.tool_call_schema.schema()
print(result)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_context]

tool_node = ToolNode(tools)

checkpointer = MemorySaver()
graph = create_react_agent(model, tools, checkpointer=checkpointer, store=doc_store)

messages = [{"type": "user", "content": "what's the latest news about FooBar"}]
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
for chunk in graph.stream({"messages": messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

messages = [{"type": "user", "content": "what's the latest news about FooBar"}]
config = {"configurable": {"thread_id": "2", "user_id": "2"}}
for chunk in graph.stream({"messages": messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
