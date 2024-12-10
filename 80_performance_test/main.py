import asyncio
import time
from functools import wraps
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import base_url, OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic.alias_generators import to_camel, to_snake

from fastapi import FastAPI, Query


load_dotenv()


def elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print("Sync execution time:", (end_time - start_time))
        return result

    return wrapper


def elapsed_time_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        print("Async execution time:", (end_time - start_time))
        return result

    return wrapper


app = FastAPI()


# 1160
@app.get("/request/sync")
def request():
    return "ok"


# 1310
@app.get("/request/async")
async def request():
    return "ok"


template = ChatPromptTemplate.from_template("test template: {question}")
model = ChatOpenAI(temperature=0, model="gpt-4o-mini", base_url="http://localhost:7777")

client = OpenAI(api_key="xxx", base_url="http://localhost:7777")
client_async = AsyncOpenAI(api_key="xxx", base_url="http://localhost:7777")


# 130 ~ 174
@app.get("/request/openai/sync")
def openai():
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "한국의 수도는?"}],
        stream=False,
    )

    return completion.choices[0].message.content


@app.get("/request/openai/async")
async def openai():
    completion = await client_async.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "한국의 수도는?"}],
        stream=False,
    )

    return completion.choices[0].message.content


# 133 ~ 160
@app.get("/request/langchain/sync")
def request():
    chain = template | model
    result = chain.invoke({"question": "테스트 질문"})
    return result


# 115
@app.get("/request/langchain/async")
async def request():
    chain = template | model
    result = await chain.ainvoke({"question": "테스트 질문"})
    return result


@app.get("/async_sync")
@elapsed_time_async
async def async_sync():  # processed sequentially
    print("start")
    time.sleep(3)  # Blocking I/O Operation, cannot be await
    # Function execution cannot be paused
    print("end")
    return "ok"


@app.get("/async_async")
@elapsed_time_async
async def async_async():  # processed concurrently
    print("start")
    await asyncio.sleep(3)  # Non-blocking I/O Operation
    # Function execution paused
    print("end")
    return "ok"


@app.get("/sync")
@elapsed_time
def sync_sync():  # processed concurrently
    print("start")
    time.sleep(3)
    print("end")
    return "ok"
