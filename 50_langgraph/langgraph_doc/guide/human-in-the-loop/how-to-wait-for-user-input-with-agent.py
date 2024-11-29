from IPython.display import Image, display
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import MessagesState, START
from langgraph.prebuilt import ToolNode

load_dotenv()


@tool
def search(query: str):
    """Call to surf the web."""
    return f"찾아봤습니다: {query}. 결과: 서울 날씨는 좋아요~ 😈."


tools = [search]
tool_node = ToolNode(tools)

# Set up the model
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

from pydantic import BaseModel


class AskHuman(BaseModel):
    """Ask the human a question"""

    question: str


model = model.bind_tools(tools + [AskHuman])


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    # 도구 호출이 사람에게 물어보는 것이면 해당 노드를 반환한다.
    # 여기에 로직을 추가하여 사람이 입력해야 하는 것이 있다는 것을 시스템에 알릴 수도 있다.
    # 예를 들어, 슬랙 메시지를 보내거나 기타 등등
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    else:
        return "continue"


def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


# 사람에게 물어보는 가짜 노드를 정의한다.
def ask_human(state):
    pass


from langgraph.graph import END, StateGraph

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node("ask_human", ask_human)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        # 'tools' 라면 도구 노드를 호출한다.
        "continue": "action",
        # 사람에게 물어볼 수 있다.
        "ask_human": "ask_human",
        # 그렇지 않다면 종료한다.
        "end": END,
    },
)

# tools에서 agent로 가는 일반적인 엣지를 추가한다.
# 이것은 tools가 호출된 후 agent 노드가 다음에 호출된다는 것을 의미한다.
workflow.add_edge("action", "agent")

# 사람의 응답을 받은 후에는 다시 agent로 돌아간다.
workflow.add_edge("ask_human", "agent")

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

app = workflow.compile(checkpointer=memory, interrupt_before=["ask_human"])

display(
    Image(
        app.get_graph().draw_mermaid_png(
            output_file_path="how-to-wait-graph-state-with-agent.png"
        )
    )
)

from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(
    content="Use the search tool to ask the user where they are, then look up the weather there"
)
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

tool_call_id = app.get_state(config).values["messages"][-1].tool_calls[0]["id"]

# id를 사용하여 도구 호출을 만들고 원하는 응답을 추가한다.
tool_message = [
    {"tool_call_id": tool_call_id, "type": "tool", "content": "san francisco"}
]

# 이것은 아래와 동일하다.
# from langchain_core.messages import ToolMessage
# tool_message = [ToolMessage(tool_call_id=tool_call_id, content="san francisco")]

# 이제 상태를 업데이트한다.
# 우리는 `as_node="ask_human"`을 지정하고 있다.
# 이것은 이 노드로 이 업데이트를 적용하게 만들 것이다.
# 이것은 이후에 계속 정상적으로 진행되도록 만들 것이다.

app.update_state(config, {"messages": tool_message}, as_node="ask_human")

# 상태를 확인할 수 있다.
# 상태에는 현재 `agent` 노드가 다음에 있음을 볼 수 있다.
# 이것은 그래프를 어떻게 정의했는지에 따라 결정된다.
# 우리는 방금 트리거한 `ask_human` 노드 이후에
# `agent` 노드로 가는 엣지가 있다.
print(app.get_state(config).next)

for event in app.stream(None, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
