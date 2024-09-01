# !pip install -U langchain langchainhub langchain_openai langchain_community -q

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# API 키 정보 로드
load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert summarizer. Please summarize the following sentence.",
        ),
        (
            "user",
            "Please summarize the sentence according to the following request."
            "\nREQUEST:\n"
            "1. Summarize the main points in bullet points in Korean."
            "2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence."
            "3. Use various emojis to make the summary more interesting."
            "\n\nCONTEXT: {context}\n\nSUMMARY:",
        ),
    ]
)

# prompt = hub.pull("teddynote/summary-stuff-documents-korean")
# print(prompt)

from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/news.txt")
docs = loader.load()
print(f"문서의 수: {len(docs)}\n")
print("[메타데이터]\n")
print(docs[0].metadata)
print("\n========= [앞부분] 미리보기 =========\n")
print(docs[0].page_content[:500])


from langchain.callbacks.base import BaseCallbackHandler


class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{token}", end="", flush=True)


llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    streaming=True,
    temperature=0.01,
    callbacks=[MyCallbackHandler()],
)
chain = create_stuff_documents_chain(llm, prompt)
answer = chain.invoke({"context": docs})
