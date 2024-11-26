from dotenv import load_dotenv
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


def should_continue(state: MessagesState):
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    # 만일 함수 호출이 없다면, 끝낸다.
    if not last_message.tool_calls:
        return END
    # 만일 함수 호출이 있다면, 계속한다.
    return "action"


# 모델을 호출하는 함수를 정의한다.
def call_model(state: MessagesState):
    response = bound_model.invoke(state["messages"])
    # 리스트를 반환한다. 기존 리스트에 추가될 것이기 때문이다.
    return {"messages": response}


workflow = StateGraph(MessagesState)

# 두 개의 노드가 서로를 순환하도록 정의한다.
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# 진입점을 'agent'로 설정한다.
# 이것은 이 노드가 처음으로 호출되는 것을 의미한다.
workflow.add_edge(START, "agent")

# 조건부 엣지를 추가한다.
workflow.add_conditional_edges(
    # 우선, 시작 노드를 정의한다. 'agent'를 사용한다.
    # 이것은 'agent' 노드가 호출된 후에 호출되는 엣지를 의미한다.
    "agent",
    # 다음으로, 다음에 호출될 노드를 결정할 함수를 전달한다.
    should_continue,
    # 다음으로, 경로 맵을 전달한다. 이 엣지가 갈 수 있는 모든 노드들이다.
    ["action", END],
)

# tools에서 agent로의 일반 엣지를 추가한다.
# 이것은 tools가 호출된 후에 agent 노드가 호출된다는 것을 의미한다.
workflow.add_edge("action", "agent")

# 마지막으로, 컴파일한다!
# 이것은 LangChain Runnable로 컴파일된다.
# 이것은 다른 Runnable처럼 사용할 수 있다는 것을 의미한다.
app = workflow.compile(checkpointer=memory)

from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(content="안녕! 내 이름은 홍길동이야")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()


input_message = HumanMessage(content="내 이름이 뭐야?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

messages = app.get_state(config).values["messages"]
print(messages)

from langchain_core.messages import RemoveMessage

app.update_state(config, {"messages": RemoveMessage(id=messages[0].id)})

messages = app.get_state(config).values["messages"]
print(messages)
