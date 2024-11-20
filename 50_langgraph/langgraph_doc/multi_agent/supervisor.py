from dotenv import load_dotenv

load_dotenv()

from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

tavily_tool = TavilySearchResults(max_results=2)

# This executes code locally, which can be unsafe
repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = (
        f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    )
    return result_str


# from langchain_core.messages import HumanMessage

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    # 'next' 필드는 다음에 어느 노드로 라우팅하는지 알려준다.
    next: str


from typing import Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI

members = ["researcher", "coder"]
# 팀 supervisor는 LLM 노드이다. 다음 실행할 에이전트를 선택하고 작업이 완료되었는지를 결정한다.
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


llm = ChatOpenAI(model="gpt-4o")


def supervisor_node(state: AgentState) -> AgentState:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    next_ = response["next"]
    if next_ == "FINISH":
        next_ = END

    return {"next": next_}


from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent


research_agent = create_react_agent(
    llm, tools=[tavily_tool], state_modifier="You are a researcher. DO NOT do any math."
)


def research_node(state: AgentState) -> AgentState:
    result = research_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="researcher")
        ]
    }


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
code_agent = create_react_agent(llm, tools=[python_repl_tool])


def code_node(state: AgentState) -> AgentState:
    result = code_agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name="coder")]
    }


builder = StateGraph(AgentState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", research_node)
builder.add_node("coder", code_node)

for member in members:
    # supervisor에게 작업이 완료되었음을 항상 알려주기를 원한다.
    builder.add_edge(member, "supervisor")


# supervisor는 그래프 상태의 "next" 필드를 채워서 노드로 라우팅하거나 종료한다.
builder.add_conditional_edges("supervisor", lambda state: state["next"])
# 마지막으로 진입점을 추가한다.
builder.add_edge(START, "supervisor")

graph = builder.compile()

from IPython.display import display, Image

display(Image(graph.get_graph().draw_mermaid_png(output_file_path="./supervisor.png")))

# 예제 1
# for s in graph.stream(
#     {"messages": [("user", "42의 제곱근은 얼마야?")]}, subgraphs=True
# ):
#     print(s)
#     print("----")


# 예제 2
for s in graph.stream(
    {
        "messages": [
            (
                "user",
                "한국과 일본의 2023년 GDP를 찾아서 평균을 계산해줘",
            )
        ]
    },
    subgraphs=True,
):
    print(s)
    print("----")
