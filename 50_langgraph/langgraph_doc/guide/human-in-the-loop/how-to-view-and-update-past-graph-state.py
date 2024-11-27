# Set up the tool
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph import MessagesState, START
from langgraph.prebuilt import ToolNode

load_dotenv()


@tool
def play_song_on_spotify(song: str):
    """Play a song on Spotify"""
    return f"Successfully played {song} on Spotify!"


@tool
def play_song_on_apple(song: str):
    """Play a song on Apple Music"""
    return f"Successfully played {song} on Apple Music!"


tools = [play_song_on_apple, play_song_on_spotify]
tool_node = ToolNode(tools)


model = ChatOpenAI(model="gpt-4o-mini")
model = model.bind_tools(tools, parallel_tool_calls=False)


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

app = workflow.compile(checkpointer=memory)

from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}
input_message = HumanMessage(content="Can you play Taylor Swift's most popular song?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

print(app.get_state(config).values["messages"])

all_states = []
for state in app.get_state_history(config):
    print(state)
    all_states.append(state)
    print("--")

to_replay = all_states[2]
print(to_replay.values)
print(to_replay.next)

for event in app.stream(None, to_replay.config):
    for v in event.values():
        print(v)

# 상태에서 마지막 메시지를 가져오자
# 이것은 업데이트하려는 도구 호출이 있는 메시지이다.
last_message = to_replay.values["messages"][-1]


# 호출하는 도구를 업데이트해보자
last_message.tool_calls[0]["name"] = "play_song_on_spotify"

branch_config = app.update_state(
    to_replay.config,
    {"messages": [last_message]},
)

for event in app.stream(None, branch_config):
    for v in event.values():
        print(v)

from langchain_core.messages import AIMessage

# 상태에서 마지막 메시지를 가져오자
# 이것은 업데이트하려는 도구 호출이 있는 메시지이다.
last_message = to_replay.values["messages"][-1]

# 마지막 메시지의 ID를 가져와서 그 ID로 새 메시지를 만들어보자.
new_message = AIMessage(
    content="It's quiet hours so I can't play any music right now!", id=last_message.id
)

branch_config = app.update_state(
    to_replay.config,
    {"messages": [new_message]},
)
