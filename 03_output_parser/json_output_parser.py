import langchain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import (
    JsonOutputParser,
)
from langchain_core.pydantic_v1 import Field
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

langchain.debug = True

load_dotenv()


class Movie(BaseModel):
    running_time: str = Field(description="상영시간에 해당")
    manager_name: str = Field(description="감독 이름")
    attendance: str = Field(description="총 관객 수에 해당")
    release_date: str = Field(description="개봉일에 해당")
    country: str = Field(description="영화를 만든 국가에 해당")


parser = JsonOutputParser(pydantic_object=Movie)

template = """
Answer the user query.
{format_instructions}

{query}

"""

query = "영화 실미도'의 상영시간, 감독, 관객 수, 개봉일, 만든 국가에 대해 알려주세요"


prompt = PromptTemplate(
    template=template,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = ChatOpenAI()
chain = prompt | model | parser

result = chain.invoke({"query": query})

print(result)
