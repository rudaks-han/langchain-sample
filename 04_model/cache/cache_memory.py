import time
from functools import wraps

from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain_community.cache import InMemoryCache
from langchain_core.output_parsers import StrOutputParser
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


set_llm_cache(InMemoryCache())

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
# 서울은 대한민국의 수도입니다. 대한민국의 정치, 경제, 문화의 중심지로서 중요한 역할을 하고 있습니다.
# Sync execution time: 0.8379083819454536

invoke()
# 서울은 대한민국의 수도입니다. 대한민국의 정치, 경제, 문화의 중심지로서 중요한 역할을 하고 있습니다.
# Sync execution time: 0.002842582995072007
