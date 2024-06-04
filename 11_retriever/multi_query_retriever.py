import logging

import langchain
from dotenv import load_dotenv
from langchain.retrievers import MultiQueryRetriever
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

langchain.debug = True
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

load_dotenv()


if __name__ == "__main__":
    print("Retrieving...")
    chat = ChatOpenAI()
    embeddings = OpenAIEmbeddings()

    db = Chroma(
        persist_directory="chroma_emb",
        embedding_function=embeddings
    )

    llm = ChatOpenAI(temperature=0)

    retriever = db.as_retriever()
    docs = retriever.invoke("Python이란 무엇인가?")
    print(docs)

    print(f"as_retriever 검색 수: {len(docs)}")
    multiquery_retriever = MultiQueryRetriever.from_llm(  # MultiQueryRetriever를 언어 모델을 사용하여 초기화합니다.
        retriever=db.as_retriever(),
        llm=llm,
    )

    result = multiquery_retriever.invoke("Python은 무엇인가?")
    print(result)
