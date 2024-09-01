# partial_variables으로 함수를 전달하여 prompt_template을 생성하는 방법

from langchain_core.prompts import PromptTemplate


def get_language():
    return "파이썬"


prompt_template = PromptTemplate(
    template="{task}을 수행하는 로직을 {language}으로 작성해 줘~",
    input_variables=["task"],
    partial_variables={"language": get_language},  # partial_variables에 함수를 전달
)

prompt = prompt_template.format(task="0부터 10까지 계산")
print(prompt)
