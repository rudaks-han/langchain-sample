from langchain.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 가짜 응답 리스트 정의
fake_responses = [
    "서울은 대한민국의 수도입니다. 대한민국의 정치, 경제, 문화의 중심지로서 중요한 역할을 하고 있습니다. 추가로 궁금한 사항이 있으시면 말씀해 주세요!"
]


# LLMChain 생성
fake_llm = FakeListLLM(
    responses=fake_responses,
)

template = "{country}의 수도는?"
prompt = PromptTemplate.from_template(template=template)
chain = prompt | fake_llm | StrOutputParser()


response = chain.invoke({"country": "서울"})
print(response)