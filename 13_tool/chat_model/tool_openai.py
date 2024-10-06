from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

from typing_extensions import Annotated, TypedDict


class add(TypedDict):
    """Add two integers."""

    a: Annotated[int, ..., "First integer"]
    b: Annotated[int, ..., "Second integer"]


class multiply(TypedDict):
    """Multiply two integers."""

    a: Annotated[int, ..., "First integer"]
    b: Annotated[int, ..., "Second integer"]


tools = [add, multiply]


llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

query = "3 곱하기 2는 뭐고 4 더하기 5는 뭐야?"

result = llm_with_tools.invoke(query).tool_calls
print(result)
