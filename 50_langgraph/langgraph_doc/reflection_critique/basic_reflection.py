import asyncio

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_fireworks import ChatFireworks

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an essay assistant tasked with writing excellent 5-paragraph essays."
            " Generate the best essay possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
llm = ChatFireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct", max_tokens=32768
)
generate = prompt | llm

essay = ""
request = HumanMessage(
    content="어린 왕자가 현대 아동기에 왜 중요한지에 대한 에세이를 한국어로 작성해줘."
)
# for chunk in generate.stream({"messages": [request]}):
#     print(chunk.content, end="")
#     essay += chunk.content

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
            " Provide detailed recommendations, including requests for length, depth, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect = reflection_prompt | llm

reflection = ""
# for chunk in reflect.stream({"messages": [request, HumanMessage(content=essay)]}):
#     print(chunk.content, end="")
#     reflection += chunk.content

# for chunk in generate.stream(
#     {"messages": [request, AIMessage(content=essay), HumanMessage(content=reflection)]}
# ):
#     print(chunk.content, end="")

from typing import Annotated, List, Sequence
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]


async def generation_node(state: State) -> State:
    return {"messages": [await generate.ainvoke(state["messages"])]}


async def reflection_node(state: State) -> State:
    # Other messages we need to adjust
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    res = await reflect.ainvoke(translated)
    # We treat the output of this as human feedback for the generator
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(State)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_edge(START, "generate")


def should_continue(state: State):
    if len(state["messages"]) > 6:
        # End after 3 iterations
        return END
    return "reflect"


builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}


async def run():
    async for event in graph.astream(
        {
            "messages": [
                HumanMessage(
                    content="어린 왕자가 현대 아동기에 왜 중요한지에 대한 에세이를 한국어로 작성해줘."
                )
            ],
        },
        config,
    ):
        print(event)
        print("---")


asyncio.run(run())

# state = graph.get_state(config)

# ChatPromptTemplate.from_messages(state.values["messages"]).pretty_print()
