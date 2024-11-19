import asyncio

from dotenv import load_dotenv

load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=3)]

from langchain import hub
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent

# 사용할 프롬프트를 가져온다. 이것을 수정해도 된다.
prompt = hub.pull("ih/ih-react-agent-executor")
prompt.pretty_print()

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-4o-mini")
agent_executor = create_react_agent(llm, tools, state_modifier=prompt)

# result = agent_executor.invoke({"messages": [("user", "한국시리즈 우승팀은 누구야?")]})
# print(result)

import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o-mini", temperature=0
).with_structured_output(Plan)

# result = planner.invoke(
#     {"messages": [("user", "현재 프로야구 우승팀 감독의 고향은 어디야?")]}
# )
# print(result)

from typing import Union


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)

from typing import Literal
from langgraph.graph import END


async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


from langgraph.graph import StateGraph, START

workflow = StateGraph(PlanExecute)

# plan node 추가
workflow.add_node("planner", plan_step)

# execution step 추가
workflow.add_node("agent", execute_step)

# replan node 추가
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")

# plan에서 agent로 연결
workflow.add_edge("planner", "agent")

# agent에서 replan으로 연결
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # 다음으로 어떤 노드를 호출할지 결정하는 함수를 전달한다.
    should_end,
    ["agent", END],
)

# 마지막으로, 컴파일한다.
# 이것은 LangChain Runnable로 컴파일된다.
# 이것은 다른 runnable처럼 사용할 수 있다.
app = workflow.compile()

from IPython.display import Image, display

display(
    Image(
        app.get_graph(xray=True).draw_mermaid_png(
            output_file_path="./plan_and_execute.png"
        )
    )
)

config = {"recursion_limit": 50}
inputs = {"input": "2023년 호주 오픈 우승자의 고향은 어디야?"}


async def run():
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


asyncio.run(run())
