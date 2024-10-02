from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import (
    JsonOutputParser,
)
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(temperature=0)


query = "한국의 수도는?"
parser = JsonOutputParser()
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser


def invoke():
    response = chain.invoke({"query": query})
    print(response)


invoke()


def stream():
    for s in chain.stream({"query": query}):
        print(s)


# stream()
