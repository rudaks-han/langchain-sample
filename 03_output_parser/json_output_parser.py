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


class Topic(BaseModel):
    description: str = Field(description="Concise description about topic")
    hashtags: str = Field(description="Some keywords in hashtag format")


# 질의 작성
query = "온난화에 대해 알려주세요."

# 파서를 설정하고 프롬프트 템플릿에 지시사항을 주입합니다.
parser = JsonOutputParser(pydantic_object=Topic)

prompt = PromptTemplate(
    # 사용자 쿼리에 답하십시오.
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],  # 입력 변수 설정
    # 부분 변수에 형식 지시사항 설정
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = ChatOpenAI(temperature=0)
chain = prompt | model | parser  # 체인을 구성합니다.

result = chain.invoke({"query": query})  # 체인을 호출하여 쿼리 실행

print(result)
