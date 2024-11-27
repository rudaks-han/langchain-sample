import asyncio
from typing import Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

load_dotenv()

# Add messages essentially does this with more
# robust handling
# def add_messages(left: list, right: list):
#     return left + right


class State(TypedDict):
    messages: Annotated[list, add_messages]


from langchain_core.tools import tool


@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    return ["The answer to your question lies within."]


tools = [search]

from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

model = model.bind_tools(tools)

from typing import Literal


# Define the function that determines whether to continue or not
def should_continue(state: State) -> Literal["end", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
async def call_model(state: State):
    messages = state["messages"]
    response = await model.ainvoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


from langgraph.graph import END, StateGraph, START

# Define a new graph
workflow = StateGraph(State)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

from IPython.display import Image, display

display(
    Image(
        app.get_graph().draw_mermaid_png(
            output_file_path="how-to-run-graph-asynchronously.png"
        )
    )
)

from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}


async def run():
    result = await app.ainvoke(inputs)
    print(result)


# asyncio.run(run())

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}


async def run():
    async for output in app.astream(inputs, stream_mode="updates"):
        # stream_mode="updates" yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value["messages"][-1].pretty_print())
        print("\n---\n")


# asyncio.run(run())

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}


async def run():
    async for output in app.astream_log(inputs, include_types=["llm"]):
        # astream_log() yields the requested logs (here LLMs) in JSONPatch format
        for op in output.ops:
            if op["path"] == "/streamed_output/-":
                # this is the output from .stream()
                ...
            elif op["path"].startswith("/logs/") and op["path"].endswith(
                "/streamed_output/-"
            ):
                # because we chose to only include LLMs, these are LLM tokens
                try:
                    content = op["value"].content[0]
                    if "partial_json" in content:
                        print(content["partial_json"], end="|")
                    elif "text" in content:
                        print(content["text"], end="|")
                    else:
                        print(content, end="|")
                except:
                    pass


asyncio.run(run())
