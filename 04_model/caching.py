from dotenv import load_dotenv
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

# langchain.debug = True
load_dotenv()

# llm = ChatOpenAI()
# set_llm_cache(InMemoryCache())
# question = "서초역 맛집을 100글자 내로 추천해줘"
# result = llm.invoke(question)
# print("첫 번째 호출")
# print(result)
#
# print("두 번째 호출")
# result = llm.invoke(question)
# print(result)


llm = ChatOpenAI()
# set_llm_cache(InMemoryCache())
set_llm_cache(SQLiteCache(database_path="llm_cache.db"))

with get_openai_callback() as callback:
    question = "서초역 맛집을 100글자 내로 추천해줘"
    print("첫 번째 호출")
    result = llm.invoke(question)
    # print(result)
    print("Total Tokens:", callback.total_tokens)

with get_openai_callback() as callback:
    question = "서초역 맛집을 100글자 내로 추천해줘"
    print("두 번째 호출")
    result = llm.invoke(question)
    # print(result)
    print("Total Tokens:", callback.total_tokens)


def sqlite_caching():
    set_llm_cache(SQLiteCache(database_path="my_llm_cache.db"))

    llm = ChatOpenAI()

    prompt = PromptTemplate.from_template("{country} 에 대해서 20자 내외로 요약해줘")

    chain = prompt | llm

    response = chain.invoke({"country": "한국"})
    print(response.content)
