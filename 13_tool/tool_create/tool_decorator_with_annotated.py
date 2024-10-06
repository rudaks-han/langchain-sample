from typing import Annotated, List

from langchain_core.tools import tool


@tool
def multiply_by_max(
    a: Annotated[str, "scale factor"],
    b: Annotated[List[int], "list of ints over which to take maximum"],
) -> int:
    """Multiply a by the maximum of b."""
    return a * max(b)


result = multiply_by_max.args_schema.schema()
print(result)
