from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticToolsParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class add(BaseModel):
    """Add two integers."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class multiply(BaseModel):
    """Multiply two integers."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


tools = [add, multiply]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)
query = "3 곱하기 2는 뭐고 4 더하기 5는 뭐야?"

chain = llm_with_tools | PydanticToolsParser(tools=[add, multiply])
result = chain.invoke(query)
print(result)
