import asyncio
import json
import os
from typing import Any

import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import LLMResult
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel

app = FastAPI()

# initialize the agent (we need to do this for the callbacks)
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
    model_name="gpt-3.5-turbo",
    streaming=True,  # ! important
    callbacks=[],  # ! important (but we will add them later)
)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True, output_key="output"
)
agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[],
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    return_intermediate_steps=False,
)


class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False

    def __init__(self) -> None:
        super().__init__()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token
        # if we passed the final answer, we put tokens in queue
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ['"', "}"]:
                    self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""


from agent_exec import get_agent_executor


async def run_call(query: str, stream_it: AsyncCallbackHandler):
    print("__ run_call")
    agent.agent.llm_chain.llm.callbacks = [stream_it]
    # await agent.acall(inputs={"input": query})
    await agent.ainvoke(inputs={"input": query})

    # agent_executor = get_agent_executor()
    # agent_executor.llm_chain.llm.callbacks = [stream_it]
    # agent_executor.callbacks = [stream_it]
    # await agent_executor(inputs={"input": query})


# request input format
class Query(BaseModel):
    text: str


async def create_gen(query: str, stream_it: AsyncCallbackHandler):
    task = asyncio.create_task(run_call(query, stream_it))
    async for token in stream_it.aiter():
        print("token", token)
        yield token
    await task


# @app.post("/chat")
# async def chat(
#     query: Query = Body(...),
# ):
#     stream_it = AsyncCallbackHandler()
#     gen = create_gen(query.text, stream_it)
#     return StreamingResponse(gen, media_type="text/event-stream")


# @app.post("/chat")
# async def chat(
#     query: Query = Body(...),
# ):
#     stream_it = AsyncCallbackHandler()
#     gen = create_gen(query.text, stream_it)
#     return StreamingResponse(gen, media_type="text/event-stream")


agent_executor = get_agent_executor()


@app.post("/chat")
async def chat(
    query: Query = Body(...),
):
    async def event_stream():
        try:

            async for event in agent_executor.astream_events(
                {"input": "where is the cat hiding? what items are in that location?"},
                version="v1",
            ):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        print(content, end="")
                        if len(content) > 0:
                            yield f"data: {json.dumps({'content': content})}\n\n"

        except Exception as e:
            # yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    """Check the api is running"""
    return {"status": "🤙"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8001, reload=True)
