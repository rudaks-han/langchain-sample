from typing import Tuple

from dotenv import load_dotenv
from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool(response_format="content_and_artifact")
def multiply(a: int, b: int) -> Tuple[str, int]:
    """Multiply two numbers."""
    content = f"{a} 곱하기 {b}은 {a * b}이다."
    artifact = a * b
    return content, artifact


# result = multiply.invoke({"a": 2, "b": 3})
# print(result)


class Multiply(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers."
    response_format: str = "content_and_artifact"

    def _run(self, a: int, b: int) -> Tuple[str, int]:
        content = f"{a} 곱하기 {b}은 {a * b}이다."
        artifact = a * b
        return content, artifact

    # 비동기 함수를 정의하려면 아래와 같이 사용
    # async def _arun(self, a: int, b: int) -> Tuple[str, int]:
    #     ...


llm = ChatOpenAI(model="gpt-4o-mini")
tools = [multiply]
llm_with_tools = llm.bind_tools(tools)
ai_msg = llm_with_tools.invoke("4 곱하기 3은 얼마인가?")
print(ai_msg.tool_calls)

result = multiply.invoke(ai_msg.tool_calls[0])
print("tool_calls", result)

result = multiply.invoke(ai_msg.tool_calls[0]["args"])
print("tool_calls args", result)

from operator import attrgetter

chain = llm_with_tools | attrgetter("tool_calls") | multiply.map()

result = chain.invoke("4 곱하기 3은 얼마인가?")
print(result)

# BaseTool로 생성
multiply = Multiply()
result = multiply.invoke({"a": 4, "b": 3})
print(result)

result = multiply.invoke(
    {
        "name": "multiply",
        "args": {"a": 4, "b": 3},
        "id": "123",
        "type": "tool_call",
    }
)
print(result)
