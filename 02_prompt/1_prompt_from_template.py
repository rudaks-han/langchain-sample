# from_template을 통해 한번에 prompt 만드는 방법

from langchain_core.prompts import PromptTemplate

template = "{task}을 수행하는 로직을 {language}으로 작성해 줘~"

prompt_template = PromptTemplate.from_template(template)
prompt = prompt_template.format(task="0부터 10까지 계산", language="파이썬")

if __name__ == "__main__":
    print(prompt)  # 0부터 10까지 계산을 수행하는 로직을 파이썬으로 작성해 줘~
