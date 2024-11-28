from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END

load_dotenv()

memory = MemorySaver()


# summary 속성을 추가한다. (MessagesState가 이미 가지고 있는 `messages` 키에 추가로)
class State(MessagesState):
    summary: str


model = ChatOpenAI(model_name="gpt-4o-mini")


# 모델을 호출하는 로직을 정의한다.
def call_model(state: State):
    # 요약이 있다면, 시스템 메시지로 추가한다.
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)

    return {"messages": [response]}


# 대화를 요약할지 끝낼지를 결정하는 로직을 정의한다.
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""
    messages = state["messages"]
    # 6개 이상의 메시지가 있다면, 대화를 요약한다.
    if len(messages) > 6:
        return "summarize_conversation"
    # 그렇지 않다면, 끝낸다.
    return END


def summarize_conversation(state: State):
    # 우선, 대화를 요약한다.
    summary = state.get("summary", "")
    if summary:
        # 이미 요약이 있다면, 요약하기 위해 다른 시스템 프롬프트를 사용한다.
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above in Korean:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    # 더 이상 표시하고 싶지 않은 메시지를 삭제해야 한다.
    # 지난 두 메시지를 제외한 모든 메시지를 삭제할 것이다. 하지만 변경할 수도 있다.
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


workflow = StateGraph(State)

# 대화 노드와 요약 노드를 정의한다.
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# 대화를 시작하는 노드로 설정한다.
workflow.add_edge(START, "conversation")

# 조건부 엣지를 추가한다.
workflow.add_conditional_edges(
    # 우선, 시작 노드를 정의한다. 'conversation'을 사용한다.
    # 'conversation' 노드가 호출된 후에 호출되는 엣지를 의미한다.
    "conversation",
    # 다음으로, 다음에 호출될 노드를 결정할 함수를 전달한다.
    should_continue,
)

# summarize_conversation에서 END로의 일반 엣지를 추가한다.
# 이것은 summarize_conversation이 호출된 후에 끝난다는 것을 의미한다.
workflow.add_edge("summarize_conversation", END)

app = workflow.compile(checkpointer=memory)


def print_update(update):
    for k, v in update.items():
        for m in v["messages"]:
            m.pretty_print()
        if "summary" in v:
            print(v["summary"])


from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "4"}}
input_message = HumanMessage(content="안녕! 내 이름은 홍길동이야")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="내 이름이 뭐야?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="나는 토트넘을 좋아해")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

values = app.get_state(config).values
print(values)

input_message = HumanMessage(content="나는 그들이 경기를 이기는 것을 좋아해")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

values = app.get_state(config).values
print(values)

input_message = HumanMessage(content="내 이름이 뭐야?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="내가 어느 야구팀을 좋아한다고 생각하니?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="나는 삼성 라이온즈를 좋아해")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
