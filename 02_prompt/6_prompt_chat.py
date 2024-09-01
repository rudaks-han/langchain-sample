from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 맛집 전문가이다."),
        ("ai", "나는 {city}을 방문하려고 한다."),
        ("human", "{question}"),
    ]
)


prompt = prompt_template.format_messages(city="서울", question="맛집 추천해줘")
print(
    prompt
)  # [SystemMessage(content='당신은 맛집 전문가이다.'), AIMessage(content='나는 서울을 방문하려고 한다.'), HumanMessage(content='맛집 추천해줘')]

prompt = prompt_template.invoke({"city": "서울", "question": "맛집 추천해줘"})
print(
    prompt
)  # messages=[SystemMessage(content='당신은 맛집 전문가이다.'), AIMessage(content='나는 서울을 방문하려고 한다.'), HumanMessage(content='맛집 추천해줘')]
