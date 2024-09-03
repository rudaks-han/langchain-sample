import langchain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

langchain.debug = False

load_dotenv()


class ResponseModel(BaseModel):
    """Response to answer user."""

    question: str = Field(description="User question")
    result: str = Field(description="The answer to the user query")


template = """
Answer the user query.

{query}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["query"],
)

model = ChatOpenAI()
structured_llm = model.with_structured_output(ResponseModel)
chain = prompt | structured_llm

query = "한국의 수도는?"
# result = chain.invoke({"query": query})
# print(result)
for chunk in chain.stream({"query": query}):
    print(chunk)
