from typing import List

from dotenv import load_dotenv

from langgraph.prebuilt.chat_agent_executor import AgentState

load_dotenv()


class State(AgentState):
    docs: List[str]


from typing_extensions import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState


@tool
def get_context(question: str, state: Annotated[dict, InjectedState]):
    """Get relevant context for answering the question."""
    return "\n\n".join(doc for doc in state["docs"])


result = get_context.get_input_schema().schema()
print(result)

result = get_context.tool_call_schema.schema()
print(result)

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_context]

tool_node = ToolNode(tools)

checkpointer = MemorySaver()
graph = create_react_agent(model, tools, state_schema=State, checkpointer=checkpointer)

docs = [
    "FooBar company just raised 1 Billion dollars!",
    "FooBar company was founded in 2019",
]

inputs = {
    "messages": [{"type": "user", "content": "what's the latest news about FooBar"}],
    "docs": docs,
}
config = {"configurable": {"thread_id": "1"}}
for chunk in graph.stream(inputs, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
