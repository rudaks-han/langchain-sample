import langchain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import (
    JsonOutputParser,
    SimpleJsonOutputParser,
)
from langchain_core.pydantic_v1 import Field
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

langchain.debug = True

load_dotenv()


def raw_json():
    template = """
    Answer the user query.
    Please provide the following information in JSON format:
    format instructions: {format_instructions}
    
    {query}
    """

    format_instructions = """
    {
        "answer": "The answer to the user query",
    }
    """
    query = "한국의 수도는?"

    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions},
    )

    model = ChatOpenAI()
    chain = prompt | model

    result = chain.invoke({"query": query})

    print("result", result)


def json_output_parser():
    template = """
    Answer the user query.
    format instructions: {format_instructions}
    
    {query}
    """

    format_instructions = """
    {
        "answer": "The answer to the user query",
    }
    """
    query = "한국의 수도는?"

    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions},
    )

    model = ChatOpenAI()
    chain = prompt | model | JsonOutputParser()

    result = chain.invoke({"query": query})

    print("result", result)


def json_mode():
    template = """
    Answer the user query.
    You must always output a JSON object with an "answer" key.
    
    {query}
    """

    query = "한국의 수도는?"

    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
    )

    model = ChatOpenAI(
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    chain = prompt | model | SimpleJsonOutputParser()

    result = chain.invoke({"query": query})

    print("result", result)


def tool_calling():
    class ResponseFormatter(BaseModel):
        """Always use this tool to structure your response to the user."""

        answer: str = Field(description="The answer to the user's question")
        followup_question: str = Field(
            description="A followup question the user could ask"
        )

    template = """
    Answer the user query.
    
    {query}
    """

    query = "한국의 수도는?"

    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
    )

    model = ChatOpenAI()
    model_with_tools = model.bind_tools([ResponseFormatter])
    chain = prompt | model_with_tools
    result = chain.invoke({"query": query})
    print("result", result)
    print("result", result.additional_kwargs["tool_calls"][0]["function"]["arguments"])


def with_structured_output():
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
    result = chain.invoke({"query": query})
    print(result)


if __name__ == "__main__":
    # raw_json()
    # json_output_parser()
    # json_mode()
    # tool_calling()
    with_structured_output()
    pass
