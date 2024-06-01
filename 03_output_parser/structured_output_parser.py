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
        description="사용자의 질문에 답하기 위해 사용된 출처를 표시해야 한다.",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI()

chain = prompt | model | output_parser

question = "2002년 한일 월드컵에서 대한민국의 성적은?"
result = chain.invoke({"question": question})
print(result)
#
# for s in chain.stream({"question": question}):
#     print(f"> {s}")
