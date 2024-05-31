from typing import List

from dotenv import load_dotenv
from langchain.output_parsers import (
    PydanticOutputParser,
)
from langchain_core.pydantic_v1 import Field
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# langchain.debug = True

load_dotenv()


class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")


actor_query = "Generate the filmography for a random actor."

parser = PydanticOutputParser(pydantic_object=Actor)

# 잘못된 형식을 일부러 입력
misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"

# 잘못된 형식으로 입력된 데이터를 파싱하려고 시도
parser.parse(misformatted)

from langchain.output_parsers import OutputFixingParser

new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())

print(misformatted)

# OutputFixingParser 를 사용하여 잘못된 형식의 출력을 파싱
actor = new_parser.parse(misformatted)

print(actor)
