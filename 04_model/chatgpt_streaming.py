from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    temperature=0,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4o",  # 모델명
    # streaming=False,
)

template = "{country}의 수도는?"

prompt = PromptTemplate.from_template(template=template)
chain = prompt | llm | StrOutputParser()


def invoke():
    response = chain.invoke({"country": "서울"})
    print(response)


# invoke()


async def ainvoke():
    response = await chain.ainvoke({"country": "서울"})
    print(response)


# asyncio.run(ainvoke())


def stream():
    for chunk in chain.stream({"country": "서울"}):
        print(chunk, end="", flush=True)


# stream()


async def astream():
    async for chunk in chain.astream({"country": "서울"}):
        print(chunk, end="", flush=True)


# asyncio.run(astream())

# chain = prompt | llm | {"result": StrOutputParser()}


async def astream_json():
    async for chunk in chain.astream({"country": "서울"}):
        print(chunk)


# asyncio.run(astream_json())
