import time
from functools import wraps

from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_elasticsearch import ElasticsearchCache
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


llm_cache = ElasticsearchCache(
    es_url="http://172.16.120.203:9200",
    index_name="llm_cache",
)

set_llm_cache(llm_cache)

llm = ChatOpenAI(
    temperature=0,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4o",  # 모델명
)

template = "{country}의 수도는?"

prompt = PromptTemplate.from_template(template=template)
chain = prompt | llm | StrOutputParser()


@elapsed_time
def invoke():
    response = chain.invoke({"country": "서울"})
    print(response)


invoke()