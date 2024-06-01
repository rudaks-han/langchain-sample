import langchain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
)
from langchain_openai import ChatOpenAI

langchain.debug = True

load_dotenv()

# 콤마로 구분된 리스트 출력 파서 초기화
output_parser = CommaSeparatedListOutputParser()

# 출력 형식 지침 가져오기
format_instructions = output_parser.get_format_instructions()
# 프롬프트 템플릿 설정
prompt = PromptTemplate(
    template="{subject}에 대해서 5개를 나열.\n{format_instructions}",
    input_variables=["subject"],  # 입력 변수로 'subject' 사용
    # 부분 변수로 형식 지침 사용
    partial_variables={"format_instructions": format_instructions},
)

# ChatOpenAI 모델 초기화
model = ChatOpenAI(temperature=0)

# 프롬프트, 모델, 출력 파서를 연결하여 체인 생성
chain = prompt | model | output_parser

question = "서초역 맛집"
result = chain.invoke({"subject": question})
print(result)

# for s in chain.stream({"subject": question}):
#     print(f"> {s}")
