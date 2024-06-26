import os
from datetime import datetime, timedelta

import faiss
from dotenv import load_dotenv
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")


def low_rate():
    # 임베딩 모델 정의
    embeddings = OpenAIEmbeddings()
    # 벡터 스토어 초기화
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings, index, InMemoryDocstore({}), {})
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, decay_rate=0.0000000000000000000000001, k=2
    )

    today = datetime.now()
    yesterday = today - timedelta(days=1)

    print("today:", today)
    print("yesterday:", yesterday)
    retriever.add_documents(
        [
            Document(
                page_content="hello world yesterday",
                metadata={"last_accessed_at": yesterday},
            )
        ]
    )
    # "hello foo" 내용의 문서를 추가합니다.
    retriever.add_documents(
        [
            Document(
                page_content="hello world today", metadata={"last_accessed_at": today}
            )
        ]
    )

    # Hello World는 가장 중요하기 때문에 먼저 반환되며, 감쇠율이 0에 가까워 여전히 최근이기 때문입니다.
    docs = retriever.invoke("hello world")
    print("====== low_rate retriever result ========")
    for i, doc in enumerate(docs):
        print(doc.metadata)
        print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")


def high_rate():
    # 임베딩 모델 정의
    embeddings_model = OpenAIEmbeddings()
    # 벡터 스토어 초기화
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, decay_rate=0.999, k=2
    )

    yesterday = datetime.now() - timedelta(days=1)
    today = datetime.now()

    retriever.add_documents(
        [
            Document(
                page_content="hello world yesterday",
                metadata={"last_accessed_at": yesterday},
            )
        ]
    )
    # "hello foo" 내용의 문서를 추가합니다.
    retriever.add_documents(
        [
            Document(
                page_content="hello world today", metadata={"last_accessed_at": today}
            )
        ]
    )

    # Hello Foo가 먼저 반환됩니다. 왜냐하면 "hello world"가 대부분 잊혀졌기 때문입니다.
    docs = retriever.invoke("hello world")
    print("====== high_rate retriever result ========")
    for i, doc in enumerate(docs):
        print(doc.metadata)
        print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")


if __name__ == "__main__":
    low_rate()
    # high_rate()
