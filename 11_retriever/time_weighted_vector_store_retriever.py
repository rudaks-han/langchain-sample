from datetime import datetime, timedelta

import faiss
from dotenv import load_dotenv
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def low_rate():
    # 임베딩 모델 정의
    embeddings = OpenAIEmbeddings()
    # 벡터 스토어 초기화
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings, index, InMemoryDocstore({}), {})
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, decay_rate=0.0000000000000000000000001, k=1
    )

    yesterday = datetime.now() - timedelta(days=1)
    retriever.add_documents(
        # "hello world" 내용의 문서를 추가하고, 메타데이터에 어제 날짜를 설정합니다.
        [Document(page_content="hello world", metadata={"last_accessed_at": yesterday})]
    )
    # "hello foo" 내용의 문서를 추가합니다.
    retriever.add_documents([Document(page_content="hello foo")])

    # Hello World는 가장 중요하기 때문에 먼저 반환되며, 감쇠율이 0에 가까워 여전히 최근이기 때문입니다.
    docs = retriever.get_relevant_documents("hello world")
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
        vectorstore=vectorstore, decay_rate=0.9999999, k=1
    )

    yesterday = datetime.now() - timedelta(days=1)
    retriever.add_documents(
        # "hello world" 내용의 문서를 추가하고, 메타데이터에 어제 날짜를 설정합니다.
        [Document(page_content="hello world", metadata={"last_accessed_at": yesterday})]
    )
    # "hello foo" 내용의 문서를 추가합니다.
    retriever.add_documents([Document(page_content="hello world")])

    docs = retriever.get_relevant_documents("hello foo")
    print("====== high_rate retriever result ========")
    for i, doc in enumerate(docs):
        print(doc.metadata)
        print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")


if __name__ == "__main__":
    low_rate()
    high_rate()
