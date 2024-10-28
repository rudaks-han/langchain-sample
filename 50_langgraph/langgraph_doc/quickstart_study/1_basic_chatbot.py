import json
from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

import logging

logging.basicConfig(level=logging.DEBUG)


class State(TypedDict):
    # add_messages 함수는 langgraph에 있는 내장 함수이며, 기존 메시지가 있으면 update하고 없으면 새로 추가한다.
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-3.5-turbo")


def chatbot(state: State):
    # State의 messages를 가져와서 llm.invoke 함수를 통해 챗봇의 응답을 생성한다.
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png(output_file_path="./chatbot.png")))
except Exception:
    # This requires some extra dependencies and is optional
    pass


def stream_graph_updates(user_input: str):
    messages = []
    for event in graph.stream({"messages": [("user", user_input)]}):
        # event: {'chatbot': {'messages': [AIMessage(content='한국의 수도는 서울입니다.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 12, 'prompt_tokens': 19, 'total_tokens': 31, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-64883620-2245-4a64-bbf0-4559bac51b23-0', usage_metadata={'input_tokens': 19, 'output_tokens': 12, 'total_tokens': 31, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}})]}}
        for value in event.values():
            messages.append(value["messages"][-1].content)

    return "\n".join(messages)


question = "한국의 수도는 어디야?"
result = stream_graph_updates(question)
print(result)
