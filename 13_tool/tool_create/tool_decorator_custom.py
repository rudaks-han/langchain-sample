from langchain_core.tools import tool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


print(f"name: {multiply.name}")
print(f"description: {multiply.description}")
print(f"args: {multiply.args}")
print(f"return_direct: {multiply.return_direct}")

if __name__ == "__main__":
    result = multiply.invoke({"a": 2, "b": 3})
    print(result)
