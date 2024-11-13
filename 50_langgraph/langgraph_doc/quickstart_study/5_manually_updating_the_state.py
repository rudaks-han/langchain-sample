from typing import Annotated

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["tools"])

user_input = "지금 LangGraph를 공부하고 있어. LangGraph에 대해 찾아줄 수 있어?"
config = {"configurable": {"thread_id": "1"}}
events = graph.stream({"messages": [("user", user_input)]}, config)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)
existing_message = snapshot.values["messages"][-1]
existing_message.pretty_print()

from langchain_core.messages import AIMessage, ToolMessage

answer = "LangGraph는 아주 아주 좋은 라이브러리야~~~"
new_messages = [
    # LLM API는 도구 호출과 일치하는 ToolMessage를 기대한다. 여기서 그것을 충족시킬 것이다.
    ToolMessage(content=answer, tool_call_id=existing_message.tool_calls[0]["id"]),
    # 이후에는 LLM의 응답에 직접 추가한다.
    AIMessage(content=answer),
]

new_messages[-1].pretty_print()
graph.update_state(
    config,
    # 업데이트 값. 우리의 `State`의 메시지는 "append-only"이다. 이것은 기존 상태에 추가될 것이다. 다음 섹션에서 기존 메시지를 업데이트하는 방법을 알아볼  것이다!
    {"messages": new_messages},
)

print("\n\nLast 2 messages;")
print(graph.get_state(config).values["messages"][-2:])

graph.update_state(
    config,
    {"messages": [AIMessage(content="나는 AI 전문가야")]},
    # 이 기능이 어느 노드에서 동작할 것인가?
    # 이것은 마치 이 노드가 방금 실행된 것처럼 처리될 것이다.
    as_node="chatbot",
)

from IPython.display import Image, display

try:
    display(
        Image(
            graph.get_graph().draw_mermaid_png(
                output_file_path="./manually_updating_the_state.png"
            )
        )
    )
except Exception:
    # This requires some extra dependencies and is optional
    pass

snapshot = graph.get_state(config)
print(snapshot.values["messages"][-3:])
print(snapshot.next)

user_input = "지금 LangGraph를 공부하고 있어. LangGraph에 대해 찾아줄 수 있어?"
config = {"configurable": {"thread_id": "2"}}  # 여기서는 thread_id = 2 를 사용한다
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

from langchain_core.messages import AIMessage

snapshot = graph.get_state(config)
existing_message = snapshot.values["messages"][-1]
print("Original")
print("Message ID", existing_message.id)
print(existing_message.tool_calls[0])
new_tool_call = existing_message.tool_calls[0].copy()
new_tool_call["args"]["query"] = "LangGraph에서 StateGraph를 사용하는 방법"
new_message = AIMessage(
    content=existing_message.content,
    tool_calls=[new_tool_call],
    # 중요! ID는 LangGraph가 이 메시지를 상태에 추가하는 것이 아니라 교체하는 방법으로 사용된다.
    id=existing_message.id,
)

print("Updated")
print(new_message.tool_calls[0])
print("Message ID", new_message.id)
graph.update_state(config, {"messages": [new_message]})

print("\n\nTool calls")
graph.get_state(config).values["messages"][-1].tool_calls

events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

events = graph.stream(
    {
        "messages": (
            "user",
            "내가 배운 것을 기억하니?",
        )
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
