# pipenv install beautifulsoup4
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.document_loaders import WebBaseLoader
from langchain.callbacks.base import BaseCallbackHandler


# Load some data to summarize
loader = WebBaseLoader("https://www.aitimes.com/news/articleView.html?idxno=131777")
docs = loader.load()
content = docs[0].page_content


class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)


prompt = hub.pull("teddynote/chain-of-density-korean")

# Create the chain, including
chain = (
    prompt
    | ChatOpenAI(
        temperature=0,
        model="gpt-4-turbo-preview",
        streaming=True,
        callbacks=[StreamCallback()],
    )
    | JsonOutputParser()
    | (lambda x: x[-1]["Denser_Summary"])
)

# Invoke the chain
result = chain.invoke({"ARTICLE": content})
print(result)
