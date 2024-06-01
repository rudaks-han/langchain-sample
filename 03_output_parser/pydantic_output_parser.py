from typing import List

import langchain
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

langchain.debug = True
load_dotenv()


class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")


actor_query = "송강호의 출연작"

parser = PydanticOutputParser(pydantic_object=Actor)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = ChatOpenAI()

chain = prompt | model | parser

result = chain.invoke({"query": actor_query})
print(result)

# 반환 형식
# class Person(BaseModel):
#     name: str = Field(description="person's name")
#     hometown: str = Field(description="person's hometown")
#     birthday: str = Field(description="person's birthday")
#
#
# # prompt 생성
# chat_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are an AI that provides information about historical figures.\n{format_instructions}",
#         ),
#         ("human", "Tell me about {name}"),
#     ]
# )
#
# # chat model 생성
# chat = ChatOpenAI()
#
# # output parser 생성
# output_parser = PydanticOutputParser(pydantic_object=Person)
#
# # chain 형성
# runnable = chat_prompt | chat | output_parser
#
# # chain 실행
# res = runnable.invoke(
#     {
#         "name": "소녀시대 윤아",
#         "format_instructions": output_parser.get_format_instructions(),
#     }
# )
# print(res)

# class MatchResult(BaseModel):
#     winner_team: str = Field(description="승자팀")
#     winner_team_score: str = Field(description="승자팀 점수")
#     loser_team: str = Field(description="패자팀")
#     loser_team_score: str = Field(description="패자팀 점수")
#
#
# class MatchSummary(BaseModel):
#     date: str = Field(description="date with the format YYYY.MM.DD")
#     match_result: List[MatchResult] = Field(description="경기 결과")
#
#
# llm = ChatOpenAI(temperature=0)
# parser = PydanticOutputParser(pydantic_object=MatchSummary)
#
# match_result = """
# 날짜: 2024.05.12, 패자팀: 김지한, 김동길, 패자팀 점수: 4점, 승자팀: 이수현, 이지훈, 승자팀 점수: 11점
# 날짜: 2024.05.12, 패자팀: 박상훈, 최선일, 패자팀 점수: 5점, 승자팀: 정은진, 유희영, 승자팀 점수: 11점
# 날짜: 2024.05.13, 패자팀: 홍길동, 허가을, 패자팀 점수: 4점, 승자팀: 이수현, 김지훈, 승자팀 점수: 11점
# 날짜: 2024.05.13, 패자팀: 이수현, 김지훈, 패자팀 점수: 5점, 승자팀: 박완규, 홍길동, 승자팀 점수: 11점
# 날짜: 2024.05.13, 패자팀: 박완규, 홍길동, 패자팀 점수: 5점, 승자팀: 김지훈, 허가을, 승자팀 점수: 11점
# 날짜: 2024.05.14, 패자팀: 이수현, 이지훈, 패자팀 점수: 4점, 승자팀: 김지한, 김동길, 승자팀 점수: 11점
# 날짜: 2024.05.14, 패자팀: 정은진, 유희영, 패자팀 점수: 5점, 승자팀: 이수현, 이지훈, 승자팀 점수: 11점
# """
#
# prompt = PromptTemplate.from_template(
#     """
# You are a helpful assistant. Please answer the following questions in KOREAN.
#
# QUESTION:
# {question}
#
# 경기 전적 결과:
# {match_result}
#
# FORMAT:
# {format}
# """
# )
# prompt = prompt.partial(format=parser.get_format_instructions())
#
# question = "5월 13일 경기 결과 알려줘"
#
# # print(prompt)
# chain = prompt | llm
# response = chain.invoke(
#     {
#         "match_result": match_result,
#         "question": question,
#     }
# )
# print(response.content)
