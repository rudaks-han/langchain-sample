import asyncio

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import ChatGenerationChunk
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables.config import (
    ensure_config,
    get_callback_manager_for_config,
)
from openai import AsyncOpenAI

load_dotenv()

openai_client = AsyncOpenAI()

tool = {
    "type": "function",
    "function": {
        "name": "get_items",
        "description": "Use this tool to look up which items are in the given place.",
        "parameters": {
            "type": "object",
            "properties": {"place": {"type": "string"}},
            "required": ["place"],
        },
    },
}


async def call_model(state, config=None):
    config = ensure_config(config | {"tags": ["agent_llm"]})
    callback_manager = get_callback_manager_for_config(config)
    messages = state["messages"]

    llm_run_manager = callback_manager.on_chat_model_start({}, [messages])[0]
    response = await openai_client.chat.completions.create(
        messages=messages, model="gpt-3.5-turbo", tools=[tool], stream=True
    )

    response_content = ""
    role = None

    tool_call_id = None
    tool_call_function_name = None
    tool_call_function_arguments = ""
    async for chunk in response:
        delta = chunk.choices[0].delta
        if delta.role is not None:
            role = delta.role

        if delta.content:
            response_content += delta.content
            llm_run_manager.on_llm_new_token(delta.content)

        if delta.tool_calls:
            # note: for simplicity we're only handling a single tool call here
            if delta.tool_calls[0].function.name is not None:
                tool_call_function_name = delta.tool_calls[0].function.name
                tool_call_id = delta.tool_calls[0].id

            # note: we're wrapping the tools calls in ChatGenerationChunk so that the events from .astream_events in the graph can render tool calls correctly
            tool_call_chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    additional_kwargs={"tool_calls": [delta.tool_calls[0].dict()]},
                )
            )
            llm_run_manager.on_llm_new_token("", chunk=tool_call_chunk)
            tool_call_function_arguments += delta.tool_calls[0].function.arguments

    if tool_call_function_name is not None:
        tool_calls = [
            {
                "id": tool_call_id,
                "function": {
                    "name": tool_call_function_name,
                    "arguments": tool_call_function_arguments,
                },
                "type": "function",
            }
        ]
    else:
        tool_calls = None

    response_message = {
        "role": role,
        "content": response_content,
        "tool_calls": tool_calls,
    }
    return {"messages": [response_message]}


import json
from langchain_core.callbacks import adispatch_custom_event


async def get_items(place: str) -> str:
    """Use this tool to look up which items are in the given place."""

    # this can be replaced with any actual streaming logic that you might have
    def stream(place: str):
        if "bed" in place:  # For under the bed
            yield from ["socks", "shoes", "dust bunnies"]
        elif "shelf" in place:  # For 'shelf'
            yield from ["books", "penciles", "pictures"]
        else:  # if the agent decides to ask about a different place
            yield "cat snacks"

    tokens = []
    for token in stream(place):
        await adispatch_custom_event(
            "tool_call_token_stream",
            {
                "function_name": "get_items",
                "arguments": {"place": place},
                "tool_output_token": token,
            },
            config={"tags": ["tool_call"]},
        )
        tokens.append(token)

    return ", ".join(tokens)


function_name_to_function = {"get_items": get_items}


async def call_tools(state):
    messages = state["messages"]

    tool_call = messages[-1]["tool_calls"][0]
    function_name = tool_call["function"]["name"]
    function_arguments = tool_call["function"]["arguments"]
    arguments = json.loads(function_arguments)

    function_response = await function_name_to_function[function_name](**arguments)
    tool_message = {
        "tool_call_id": tool_call["id"],
        "role": "tool",
        "name": function_name,
        "content": function_response,
    }
    return {"messages": [tool_message]}


import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START


class State(TypedDict):
    messages: Annotated[list, operator.add]


def should_continue(state) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message["tool_calls"]:
        return "tools"
    return END


workflow = StateGraph(State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)  # i.e. our "agent"
workflow.add_node("tools", call_tools)
workflow.add_conditional_edges("model", should_continue)
workflow.add_edge("tools", "model")
graph = workflow.compile()


async def stream_content():
    async for event in graph.astream_events(
        {"messages": [{"role": "user", "content": "what's in the bedroom"}]},
        version="v2",
    ):
        tags = event.get("tags", [])
        if event["event"] == "on_custom_event" and "tool_call" in tags:
            print("Tool token", event["data"]["tool_output_token"])


asyncio.run(stream_content())
