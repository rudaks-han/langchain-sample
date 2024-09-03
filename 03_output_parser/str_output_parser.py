from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
)
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(temperature=0)

template = "{task}을 수행하는 로직을 {language}으로 작성해 줘~"
prompt = PromptTemplate(template=template)
chain = prompt | model | StrOutputParser()


def invoke():
    response = chain.invoke({"task": "0부터 10까지 계산", "language": "파이썬"})
    print(response)


invoke()


def stream():
    for s in chain.stream({"query": "한국의 수도는?"}):
        print(s)


# stream()
