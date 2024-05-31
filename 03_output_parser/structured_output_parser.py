import langchain
from dotenv import load_dotenv
from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
)
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

langchain.debug = True

load_dotenv()

response_schemas = [
    ResponseSchema(name="answer", description="사용자의 질문에 대한 답변"),
    ResponseSchema(
        name="source",
        description="사용자의 질문에 답하기 위해 사용된 출처, 웹사이트 이여야 합니다.",
    ),
]
# 응답 스키마를 기반으로 한 구조화된 출력 파서 초기화
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 출력 형식 지시사항을 파싱합니다.
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    # 사용자의 질문에 최대한 답변하도록 템플릿을 설정합니다.
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    # 입력 변수로 'question'을 사용합니다.
    input_variables=["question"],
    # 부분 변수로 'format_instructions'을 사용합니다.
    partial_variables={"format_instructions": format_instructions},
)

# ChatOpenAI 모델 초기화
model = ChatOpenAI(temperature=0)

# 프롬프트, 모델, 출력 파서를 연결하여 체인 생성
chain = prompt | model | output_parser

question = "대한민국의 수도는 어디인가요?"
# result = chain.invoke({"subject": question})
# print(result)

for s in chain.stream({"question": question}):
    print(f"> {s}")
