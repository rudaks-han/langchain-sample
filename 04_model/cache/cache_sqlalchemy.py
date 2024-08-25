import time
from functools import wraps

from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain_community.cache import SQLAlchemyCache
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine

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


engine = create_engine("sqlite:///sqlalchemy_cache.db")
llm_cache = SQLAlchemyCache(engine=engine)
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
