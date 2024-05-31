from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.prompts import load_prompt

# template = "{task}을 수행하는 로직을 {language}으로 작성해 줘~"
#
# prompt_template = PromptTemplate.from_template(template)
# print(prompt_template)
# prompt = prompt_template.format(task="0부터 10까지 계산", language="파이썬")
# print(prompt)  # 0부터 10까지 계산을 수행하는 로직을 파이썬으로 작성해 줘~
# prompt_template = load_prompt("prompts/template.json")
# prompt = prompt_template.format(num1=3, num2=5, operator="+")
# print(prompt)
# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "당신은 맛집 전문가이다."),
#         ("ai", "나는 {city}을 방문하려고 한다."),
#         ("human", "{question}"),
#     ]
# )
#
# prompt = prompt_template.format_messages(city="서울", question="맛집 추천해줘")
# print(prompt)
# from langchain_core.prompts import ChatMessagePromptTemplate
#
# prompt = "나는 {country}로 여행가고 싶어"
#
# chat_message_prompt = ChatMessagePromptTemplate.from_template(
#     role="Steve", template=prompt
# )
# prompt = chat_message_prompt.format(country="한국")
# print(prompt)

human_prompt = "Summarize our conversation so far in {word_count} words."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="conversation"), human_message_template]
)

print(chat_prompt)


# 방법 1: from_template을 사용하는 방법
def task1():
    template = "{task}을 수행하는 로직을 {language}으로 작성해 줘~"

    prompt_template = PromptTemplate.from_template(template)
    prompt = prompt_template.format(task="0부터 10까지 계산", language="파이썬")
    print(prompt)  # 0부터 10까지 계산을 수행하는 로직을 파이썬으로 작성해 줘~

    # 출력: 0부터 10까지 계산을 수행하는 로직을 파이썬으로 작성해 줘~


# 방법 2: PromptTemplate 객체 생성과 동시에 prompt 생성
def task2():
    template = "{task}을 수행하는 로직을 {language}으로 작성해 줘~"
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["task", "language"],  # input_variables=[] 으로 해도 실행된다.
    )

    prompt = prompt_template.format(task="0부터 10까지 계산", language="파이썬")
    print(prompt)  # 0부터 10까지 계산을 수행하는 로직을 파이썬으로 작성해 줘~


def load_template():
    prompt_template = load_prompt("prompts/template.json")
    prompt = prompt_template.format(num1=3, num2=5, operator="+")
    print(prompt)


def get_today():
    now = datetime.now()
    return now.strftime("%Y-%m-%d")


def partial_variable():
    prompt_template = PromptTemplate(
        template="오늘의 {today} 입니다. 그리고 {n}을 입력 받았습니다.",
        input_variables=["n"],
        partial_variables={"today": get_today},  # partial_variables에 함수를 전달
    )

    prompt = prompt_template.format(n=10)
    print(prompt)  # 오늘의 2024-05-29 입니다. 그리고 10을 입력 받았습니다.


from langchain_core.runnables import RunnablePassthrough


def runnable_passthorugh():
    prompt_template = PromptTemplate(
        template="오늘의 {today} 입니다. 그리고 {n}을 입력 받았습니다.",
        input_variables=["n"],
        partial_variables={"today": get_today},  # partial_variables에 함수를 전달
    )
    runnable_template = {"n": RunnablePassthrough()} | prompt_template
    prompt = runnable_template.invoke(5)
    print(prompt)

    # 아래 로직과 동일
    # runnable_template = {"n": RunnablePassthrough()}
    # runnable_template.update(prompt_template)
    # prompt = prompt_template.invoke(5)


# if __name__ == "__main__":
# task1()
# task2()
# partial_variable()
# runnable_passthorugh()


def chat_prompt():
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 맛집 전문가이다."),
            ("ai", "나는 {city}을 방문하려고 한다."),
            ("human", "{question}"),
        ]
    )

    prompt = prompt_template.format_messages(city="서울", question="맛집 추천해줘")
    print(prompt)


def message_template():
    from langchain_core.prompts import ChatMessagePromptTemplate

    prompt = "나는 {country}로 여행가고 싶어"

    chat_message_prompt = ChatMessagePromptTemplate.from_template(
        role="Steve", template=prompt
    )
    chat_message_prompt.format(country="한국")


# 참고: https://wikidocs.net/233351
