import asyncio
import time
from functools import wraps
from time import sleep

from dotenv import load_dotenv
from langchain_core.language_models import FakeListLLM, FakeStreamingListLLM

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


# 가짜 응답 리스트 정의
fake_responses = [
    "서울은 대한민국의 수도입니다. 대한민국의 정치, 경제, 문화의 중심지로서 중요한 역할을 하고 있습니다. 추가로 궁금한 사항이 있으시면 말씀해 주세요!"
]


# LLM 호출용 FakeListLLM
llm = FakeListLLM(
    responses=fake_responses,
)
# streaming 용 FakeStreamingListLLM
llm_streaming = FakeStreamingListLLM(responses=fake_responses)

template = "한국의 수도는?"
request_count = 10
sleep_time = 0.5


@elapsed_time
def invoke_sync():
    for i in range(request_count):
        result = llm.invoke(template)
        sleep(sleep_time)
        print(result)


# invoke_sync()
# Sync execution time: 5.045806487905793


async def ainvoke():
    result = await llm.ainvoke(template)
    await asyncio.sleep(sleep_time)
    print(result)


@elapsed_time_async
async def invoke_async():
    tasks = [ainvoke() for _ in range(request_count)]
    await asyncio.gather(*tasks)


# asyncio.run(invoke_async())
# Async execution time: 0.5098811800125986


@elapsed_time
def stream_sync():
    for i in range(request_count):
        for chunk in llm_streaming.stream(template):
            print(chunk, end="", flush=True)

        print("\n", end="", flush=True)
        sleep(sleep_time)


# stream_sync()
# Sync execution time: 5.051549619063735


async def astream():
    async for chunk in llm_streaming.astream(template):
        print(chunk, end="", flush=True)

    print("\n", end="", flush=True)
    await asyncio.sleep(sleep_time)


@elapsed_time_async
async def stream_async():
    tasks = [astream() for _ in range(request_count)]
    await asyncio.gather(*tasks)


# asyncio.run(stream_async())
# Async execution time: 0.5122395370854065
