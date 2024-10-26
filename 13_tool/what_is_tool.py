import langchain
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

langchain.debug = True
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


@tool
def multiply(a: int, b: int) -> int:
    """Divide two numbers."""
    return a * b


@tool
def divide(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a / b


# result = multiply.invoke({"a": 2, "b": 3})
# result = multiply.invoke(
#     {
#         "name": "multiply",
#         "args": {"a": 2, "b": 3},
#         "id": "123",
#         "type": "tool_call",
#     }
# )
# print(result)

tools = [multiply, divide]  # 다양한 도구를 정의한다.
llm_with_tools = llm.bind_tools(tools)  # llm에 도구들을 bind한다.
ai_msg = llm_with_tools.invoke("4 곱하기 3은 얼마인가?")  # bind된 도구들을 실행한다.
# print(f"ai_msg.tool_calls: {ai_msg.tool_calls}")
# 결과: [{'name': 'multiply', 'args': {'a': 4, 'b': 3}, 'id': 'call_lYc3VnTLw1bF5f3P2KwfM9gx', 'type': 'tool_call'}]

tool_call = ai_msg.tool_calls[0]
print(f"tool_call: {tool_call}")
#
# 단순 인수로 호출
# tool_output = multiply.invoke(tool_call["args"])
# print(tool_output)
#
# tool_message = ToolMessage(
#     content=tool_output, tool_call_id=tool_call["id"], name=tool_call["name"]
# )
# print(tool_message)


# ToolCall로 호출
# tool_message = multiply.invoke(tool_call)
# print(tool_message)
