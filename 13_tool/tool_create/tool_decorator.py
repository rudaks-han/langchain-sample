from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


print(f"name: {multiply.name}")
print(f"description: {multiply.description}")
print(f"args: {multiply.args}")

if __name__ == "__main__":
    result = multiply.invoke({"a": 2, "b": 3})
    print(result)
