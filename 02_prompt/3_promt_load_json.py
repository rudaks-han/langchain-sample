# json 파일을 읽어들이는 방법

from langchain_core.prompts import load_prompt

prompt_template = load_prompt("sample/template.json")
prompt = prompt_template.format(task="0부터 10까지 계산", language="파이썬")
print(prompt)
