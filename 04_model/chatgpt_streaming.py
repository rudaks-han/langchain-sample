import asyncio

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# llm = ChatOpenAI()
llm = ChatOpenAI(
    temperature=0,  # 창의성 (0.0 ~ 2.0)
    max_tokens=1024,  # 최대 토큰수
    model_name="gpt-4o",  # 모델명
    streaming=True,
)

template = "{country}의 수도는?"

prompt = PromptTemplate.from_template(template=template)
# chain = LLMChain(prompt=prompt, llm=llm)
#
# result = chain.invoke({"country", "한국"})
# print(result)
parser = StrOutputParser()
chain = prompt | llm | parser


async def print_result():
    async for chunk in chain.astream({"country": "서울"}):
        print(chunk, end="\n", flush=True)


asyncio.run(print_result())

# async def print_result(chain):
#     async for chunk in chain.astream({"country": "서울"}):
#         # print(chunk, end="|", flush=True)
#         print(chunk)
#
#
# def main():
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(print_result(chain))
#     loop.close()
#
#
# if __name__ == "__main__":
#     main()
