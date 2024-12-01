import asyncio

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import ChatGenerationChunk
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
        messages=messages, model="gpt-4o-mini", tools=[tool], stream=True
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
            # stream_mode="messages"를 사용하여 이를 다시 스트리밍할 수 있도록 응답을 ChatGenerationChunk로 래핑한다.
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=delta.content,
                )
            )
            llm_run_manager.on_llm_new_token(delta.content, chunk=chunk)

        if delta.tool_calls:
            # 단순화하기 위해 여기서 하나의 도구 호출만 처리한다.
            if delta.tool_calls[0].function.name is not None:
                tool_call_function_name = delta.tool_calls[0].function.name
                tool_call_id = delta.tool_calls[0].id

            # stream_mode="messages"를 사용하여 이를 다시 스트리밍할 수 있도록 응답을 ChatGenerationChunk로 래핑한다.
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


async def get_items(place: str) -> str:
    """Use this tool to look up which items are in the given place."""
    if "침실" in place:
        return "양말, 신발, 먼지 덩어리"
    elif "선반" in place:
        return "책, 연필, 그림"
    else:
        return "고양이 스낵"


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
workflow.add_node("model", call_model)
workflow.add_node("tools", call_tools)
workflow.add_conditional_edges("model", should_continue)
workflow.add_edge("tools", "model")
graph = workflow.compile()

from langchain_core.messages import AIMessageChunk


async def stream_async():
    first = True
    async for msg, metadata in graph.astream(
        {"messages": [{"role": "user", "content": "침실에 뭐가 있어?"}]},
        stream_mode="messages",
    ):
        if msg.content:
            print(msg.content, end="|", flush=True)

        if isinstance(msg, AIMessageChunk):
            if first:
                gathered = msg
                first = False
            else:
                gathered = gathered + msg

            if msg.tool_call_chunks:
                print(gathered.tool_calls)


asyncio.run(stream_async())
