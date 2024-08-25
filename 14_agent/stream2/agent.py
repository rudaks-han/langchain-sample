# Get the prompt to use - you can modify this!
import asyncio
import random

from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
async def where_cat_is_hiding() -> str:
    """Where is the cat hiding right now?"""
    return random.choice(["under the bed", "on the shelf"])


@tool
async def get_items(place: str) -> str:
    """Use this tool to look up which items are in the given place."""
    if "bed" in place:  # For under the bed
        return "socks, shoes and dust bunnies"
    if "shelf" in place:  # For 'shelf'
        return "books, penciles and pictures"
    else:  # if the agent decides to ask about a different place
        return "cat snacks"


prompt = hub.pull("hwchase17/openai-tools-agent")
model = ChatOpenAI(temperature=0, streaming=True)


# print(prompt.messages) -- to see the prompt
tools = [get_items, where_cat_is_hiding]
agent = create_openai_tools_agent(
    model.with_config({"tags": ["agent_llm"]}), tools, prompt
)
agent_executor = AgentExecutor(agent=agent, tools=tools).with_config(
    {"run_name": "Agent"}
)


async def print_result():
    async for event in agent_executor.astream_events(
        # {"input": "where is the cat hiding? what items are in that location?"},
        {"input": "tell me a short story"},
        version="v1",
    ):
        kind = event["event"]
        # print("________")
        if kind == "on_chain_start":
            pass
            # if (
            #     event["name"] == "Agent"
            # ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
            #     print(
            #         f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
            #     )
        elif kind == "on_chain_end":
            pass
            # if (
            #     event["name"] == "Agent"
            # ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
            #     print()
            #     print("-- on_chain_end")
            #     print(
            #         f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
            #     )
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="")
        elif kind == "on_tool_start":
            pass
            # print("-- on_tool_start")
            # print(
            #     f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            # )
        elif kind == "on_tool_end":
            pass
            # print(f"Done tool: {event['name']}")
            # print(f"Tool output was: {event['data'].get('output')}")
            # print("--")


asyncio.run(print_result())
