# First we initialize the model we want to use.
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# For this tutorial we will use custom tool that returns pre-defined values for weather in two cities (NYC & SF)

from langchain_core.tools import tool


@tool
def get_weather(location: str):
    """Use this to get weather information from a given location."""
    if location.lower() in ["nyc", "new york"]:
        return "It might be cloudy in nyc"
    elif location.lower() in ["sf", "san francisco"]:
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown Location")


tools = [get_weather]

# We need a checkpointer to enable human-in-the-loop patterns
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Define the graph

from langgraph.prebuilt import create_react_agent

graph = create_react_agent(
    model, tools=tools, interrupt_before=["tools"], checkpointer=memory
)


def print_stream(stream):
    """A utility to pretty print the stream."""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


config = {"configurable": {"thread_id": "42"}}
inputs = {"messages": [("user", "what is the weather in SF, CA?")]}

print_stream(graph.stream(inputs, config, stream_mode="values"))

snapshot = graph.get_state(config)
print("Next step: ", snapshot.next)

print_stream(graph.stream(None, config, stream_mode="values"))

state = graph.get_state(config)

last_message = state.values["messages"][-1]
last_message.tool_calls[0]["args"] = {"location": "San Francisco"}

update_state = graph.update_state(config, {"messages": [last_message]})
print(update_state)

print_stream(graph.stream(None, config, stream_mode="values"))
