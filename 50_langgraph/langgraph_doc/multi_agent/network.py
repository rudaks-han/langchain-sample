from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph, START


def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

tavily_tool = TavilySearchResults(max_results=2)

# Warning: This executes code locally, which can be unsafe when not sandboxed

repl = PythonREPL()


@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = (
        f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    )
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI


# 여기서 각 노드에서 전달되는 객체를 정의한다.
# 그래프에서 각 에이전트와 도구에 다른 노드를 생성한다.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


import functools

from langchain_core.messages import AIMessage


# 해당 에이전트의 노드를 만드는데 필요한 헬퍼 함수
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # 에이전트 출력을 전역 상태에 추가될 형식으로 변환한다.
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # 엄격한 워크플로우를 가지고 있기 때문에 다음에 누구에게 전달하는지 알 수 있다.
        "sender": name,
    }


llm = ChatOpenAI(model="gpt-4o")

# Research 에이전트와 노드
research_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You should provide accurate data for the chart_generator to use.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# chart_generator 에이전트와 노드
chart_agent = create_agent(
    llm,
    [python_repl],
    system_message="Any charts you display will be visible by the user.",
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

from langgraph.prebuilt import ToolNode

tools = [tavily_tool, python_repl]
tool_node = ToolNode(tools)

# Either agent can decide to end


def router(state):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", END: END},
)
workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", END: END},
)

workflow.add_conditional_edges(
    "call_tool",
    # 각 에이전트는 'sender' 필드를 업데이트한다.
    # 도구를 호출하는 노드는 호출한 원래 에이전트로 돌아간다.
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "chart_generator": "chart_generator",
    },
)
workflow.add_edge(START, "Researcher")
graph = workflow.compile()

from IPython.display import Image, display

try:
    display(
        Image(
            graph.get_graph(xray=True).draw_mermaid_png(
                output_file_path="./network.png"
            )
        )
    )
except Exception:
    # This requires some extra dependencies and is optional
    pass

events = graph.stream(
    {
        "messages": [
            HumanMessage(
                # content="Fetch the UK's GDP over the past 5 years,"
                # " then draw a line graph of it."
                # " Once you code it up, finish."
                content="과거 5년 동안 한국의 GDP를 가져와서 라인 그래프로 그리고, 완료해줘~"
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 100},
)
for s in events:
    print(s)
    print("----")
