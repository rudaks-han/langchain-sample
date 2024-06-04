import langchain
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

langchain.debug = True

load_dotenv()

llm = ChatOpenAI()

with get_openai_callback() as cb:
    result = llm.invoke("대한민국의 수도는 어디야?")
    print(cb)

# with get_openai_callback() as cb:
#     result = llm.invoke("대한민국의 수도는 어디야?")
#     print(f"총 사용된 토큰수: \t\t{cb.total_tokens}")
#     print(f"프롬프트에 사용된 토큰수: \t{cb.prompt_tokens}")
#     print(f"답변에 사용된 토큰수: \t{cb.completion_tokens}")
#     print(f"호출에 청구된 금액(USD): \t${cb.total_cost}")
