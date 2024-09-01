import langchain
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_openai import ChatOpenAI

langchain.debug = True
load_dotenv()

template = ChatPromptTemplate(
    [
        ("system", "너는 AI 봇이야"),
        ("placeholder", "{conversation}"),
        ("human", "거기에 1을 더하면?"),
        # MessagesPlaceholder(variable_name="conversation", optional=True)
    ]
)

prompt = template.invoke(
    {
        "conversation": [
            ("human", "안녕~"),
            ("ai", "무엇을 도와드릴까요?"),
            ("human", "1+1은 뭐야?"),
            ("ai", "2입니다."),
        ]
    }
)

print(prompt)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

chain = template | llm | StrOutputParser()
result = chain.invoke(
    {
        "conversation": [
            ("human", "안녕~"),
            ("ai", "무엇을 도와드릴까요?"),
            ("human", "1+1은 뭐야?"),
            ("ai", "2입니다."),
        ],
    },
)

print(result)
