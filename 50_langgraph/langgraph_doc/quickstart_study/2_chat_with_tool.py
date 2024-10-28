from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.constants import END, START

load_dotenv()

tool = TavilySearchResults(max_results=2)
tools = [tool]

from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = ChatOpenAI(model="gpt-3.5-turbo", verbose=True)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)


import json

from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)


def route_tools(
    state: State,
):
    """
    마지막 메시지에 도구 호출이 있으면 ToolNode로 라우팅하기 위해 conditional_edge에서 사용한다.
    그렇지 않으면 종료로 라우팅한다.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"  # 도구를 사용해야 할 때 "tools"를 반환한다.

    return END


# tools_condition 함수는 챗봇이 도구를 사용해야 하는지 여부를 판단하여,
# 도구를 사용해야 할 때는 "tools"를 반환하고, 직접 응답해도 될 때는 "END"를 반환한다.
# 이 조건부 라우팅은 메인 에이전트 루프를 정의한다.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # 다음 dict는 그래프에 조건의 출력을 특정 노드로 해석하도록 지시할 수 있다.
    # 기본적으로는 identity 함수로 설정되지만, "tools" 이외의 다른 이름을 가진 노드를 사용하려면 dict 값을 변경할 수 있다.
    # 예: "tools": "my_tools"    {"tools": "tools", END: END},
)
# 도구가 호출될 때마다 우리는 다음 단계를 결정하기 위해 다시 챗봇으로 돌아간다.
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

from IPython.display import Image, display

try:
    display(
        Image(
            graph.get_graph().draw_mermaid_png(
                output_file_path="./chatbot_with_tool.png"
            )
        )
    )
except Exception:
    # This requires some extra dependencies and is optional
    pass


def stream_graph_updates(user_input: str):
    messages = []
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            messages.append(value["messages"][-1].content)
            print(f"messages: {messages}")

    return "\n".join(messages)


question = "오늘 서울 날씨가 어때?"
result = stream_graph_updates(question)
print(f"result: {result}")
