from langchain_core.tools import tool


@tool(parse_docstring=True)
def user(id: int, name: str) -> str:
    """The user.

    Args:
        id: user id.
        name: user name.
    """
    return f"{name} ({id})"


result = user.args_schema.schema()
print(result)
