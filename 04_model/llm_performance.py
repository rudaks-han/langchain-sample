import asyncio
import time
from functools import wraps

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

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


llm = ChatOpenAI(
    temperature=0,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4o",  # 모델명
)

template = "한국의 수도는?"
request_count = 10


@elapsed_time
def invoke_sync():
    for i in range(request_count):
        result = llm.invoke(template)
        print(result.content)


# invoke_sync()
# Sync execution time: 7.861405140021816


async def ainvoke():
    result = await llm.ainvoke(template)
    print(result.content)


@elapsed_time_async
async def invoke_async():
    tasks = [ainvoke() for _ in range(request_count)]
    await asyncio.gather(*tasks)


# asyncio.run(invoke_async())
# Async execution time: 0.7999031220097095


@elapsed_time
def stream_sync():
    for i in range(request_count):
        for chunk in llm.stream(template):
            print(chunk.content)


# stream_sync()
# Sync execution time: 6.138112443964928


async def astream():
    async for chunk in llm.astream(template):
        print(chunk.content)


@elapsed_time_async
async def stream_async():
    tasks = [astream() for _ in range(request_count)]
    await asyncio.gather(*tasks)


# asyncio.run(stream_async())
# Async execution time: 10.570893630036153
