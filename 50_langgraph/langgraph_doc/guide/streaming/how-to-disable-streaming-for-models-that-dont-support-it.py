import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(model="o1-preview", temperature=1, disable_streaming=True)

graph_builder = StateGraph(MessagesState)


def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

from IPython.display import Image, display

display(
    Image(
        graph.get_graph().draw_mermaid_png(
            output_file_path="how-to-disable-streaming-for-models-that-dont-support-it.png"
        )
    )
)

input = {"messages": {"role": "user", "content": "strawberry에 r이 몇개 있지?"}}


async def stream_content():
    try:
        async for event in graph.astream_events(input, version="v2"):
            if event["event"] == "on_chat_model_end":
                print(event["data"]["output"].content, end="", flush=True)
    except:
        print("Streaming not supported!")


asyncio.run(stream_content())
