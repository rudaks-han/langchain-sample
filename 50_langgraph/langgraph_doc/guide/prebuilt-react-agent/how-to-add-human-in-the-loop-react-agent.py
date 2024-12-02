# First we initialize the model we want to use.

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


from langchain_core.tools import tool


@tool
def get_weather(location: str):
    """Use this to get weather information from a given location."""
    if location.lower() in ["nyc", "new york"]:
        return "nyc는 흐린거 같아요"
    elif location.lower() in ["sf", "san francisco"]:
        return "sf는 항상 맑아요"
    else:
        raise AssertionError("Unknown Location")


tools = [get_weather]

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()


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
inputs = {"messages": [("user", "SF, CA 날씨 어때?")]}

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
