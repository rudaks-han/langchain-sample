import langchain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import (
    SimpleJsonOutputParser,
)
from langchain_openai import ChatOpenAI

langchain.debug = True

load_dotenv()

template = """
{country}의 수도는?
You must always output a JSON object with an "answer" key and a "followup_question" key.
"""


prompt = PromptTemplate(
    template=template,
    input_variables=["country"],
)

llm = ChatOpenAI(
    model="gpt-4o",
    model_kwargs={"response_format": {"type": "json_object"}},
)
chain = prompt | llm | SimpleJsonOutputParser()

result = chain.invoke({"country": "한국"})
print(result)
