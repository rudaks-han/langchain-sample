import operator
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict

model = ChatOpenAI(model_name="gpt-4o-mini")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def _call_model(state):
    state["messages"]
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# Define a new graph
builder = StateGraph(AgentState)
builder.add_node("model", _call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()

from langchain_openai import ChatOpenAI
from typing import Optional
from langchain_core.runnables.config import RunnableConfig

openai_model = ChatOpenAI()

models = {
    "anthropic": model,
    "openai": openai_model,
}


def _call_model(state: AgentState, config: RunnableConfig):
    # Access the config through the configurable key
    model_name = config["configurable"].get("model", "anthropic")
    model = models[model_name]
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# Define a new graph
builder = StateGraph(AgentState)
builder.add_node("model", _call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="hi")]})
print(result)

config = {"configurable": {"model": "openai"}}
result = graph.invoke({"messages": [HumanMessage(content="hi")]}, config=config)
print(result)

from langchain_core.messages import SystemMessage


# We can define a config schema to specify the configuration options for the graph
# A config schema is useful for indicating which fields are available in the configurable dict inside the config
class ConfigSchema(TypedDict):
    model: Optional[str]
    system_message: Optional[str]


def _call_model(state: AgentState, config: RunnableConfig):
    # Access the config through the configurable key
    model_name = config["configurable"].get("model", "anthropic")
    model = models[model_name]
    messages = state["messages"]
    if "system_message" in config["configurable"]:
        messages = [
            SystemMessage(content=config["configurable"]["system_message"])
        ] + messages
    response = model.invoke(messages)
    return {"messages": [response]}


# Define a new graph - note that we pass in the configuration schema here, but it is not necessary
workflow = StateGraph(AgentState, ConfigSchema)
workflow.add_node("model", _call_model)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

graph = workflow.compile()

result = graph.invoke({"messages": [HumanMessage(content="hi")]})
print(result)

config = {"configurable": {"system_message": "respond in italian"}}
result = graph.invoke({"messages": [HumanMessage(content="hi")]}, config=config)
print(result)
