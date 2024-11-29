from IPython.display import display, Image
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph import MessagesState, START
from langgraph.prebuilt import ToolNode

load_dotenv()


@tool
def search(query: str):
    """Call to surf the web."""
    return ["서울 날씨는 맑아~ 😈."]


tools = [search]
tool_node = ToolNode(tools)

model = ChatOpenAI(model="gpt-4o-mini")
model = model.bind_tools(tools)


# 진행을 계속할지 결정하는 함수
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # 함수 호출이 없다면 중단한다.
    if not last_message.tool_calls:
        return "end"
    # 함수 호출이 있다면 계속한다.
    else:
        return "continue"


# 모델을 호출하는 함수를 정의한다.
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # 리스트를 리턴한다. 기존 리스트에 추가된다.
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

# 두 노드가 서로를 순환하도록 정의한다.
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# 시작점을 'agent'로 설정한다.
# 이것은 이 노드가 처음으로 호출되는 것을 의미한다.
workflow.add_edge(START, "agent")

# 조건부 엣지를 추가한다.
workflow.add_conditional_edges(
    # 'agent' 노드가 호출된 후에 호출되는 엣지를 의미한다.
    "agent",
    # 다음으로 다음에 호출될 노드를 결정할 함수를 전달한다.
    should_continue,
    # 마지막으로 매핑을 전달한다.
    # 키는 문자열이고, 값은 다른 노드들이다.
    # END는 그래프가 끝나야 한다는 것을 표시하는 특별한 노드이다.
    # 이후 `should_continue`를 호출하고, 그 출력이 이 매핑의 키들과 일치하게 된다.
    # 매칭이 된다면 노드가 호출된다.
    {
        # 만일 `tools`이라면, 도구 노드를 호출한다.
        "continue": "action",
        # 그렇지 않다면 끝낸다.
        "end": END,
    },
)

# tools에서 agent로의 일반 엣지를 추가한다.
# 이는 tools가 호출된 후에 agent 노드가 호출된다는 것을 의미한다.
workflow.add_edge("action", "agent")

# 메모리 설정
memory = MemorySaver()


# interrupt_before=["action"]를 추가한다.
# 이것은 `action` 노드가 호출되기 전에 중단점을 추가한다.
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])

display(
    Image(
        app.get_graph().draw_mermaid_png(output_file_path="how-to-add-breakpoint2.png")
    )
)

from langchain_core.messages import HumanMessage

thread = {"configurable": {"thread_id": "3"}}
inputs = [HumanMessage(content="지금 서울 날씨 검색해 줘")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

for event in app.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
