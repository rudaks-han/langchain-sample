from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import utils as chromautils
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

# 임베딩 모델 정의
embeddings = OpenAIEmbeddings()

vector_store = Chroma(
    persist_directory="chroma_time_weighted",
    collection_name="full_documents",
    embedding_function=embeddings,
)

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vector_store, decay_rate=0.0000000000000000000000001, k=2
)

today = datetime.now()
yesterday = today - timedelta(days=1)

docs = [
    Document(
        page_content="hello world yesterday",
        metadata={"last_accessed_at": yesterday, "created_at": today},
    ),
    # Document(
    #     page_content="hello world today",
    #     metadata={"last_accessed_at": str(today), "created_at": str(today)},
    # ),
]
docs = chromautils.filter_complex_metadata(docs)


retriever.add_documents(docs, ids=None, add_to_docstore=True)

# Hello World는 가장 중요하기 때문에 먼저 반환되며, 감쇠율이 0에 가까워 여전히 최근이기 때문입니다.
docs = retriever.invoke("hello world")
print("====== low_rate retriever result ========")
for i, doc in enumerate(docs):
    print(doc.metadata)
    print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")


def low_rate():

    # 벡터 스토어 초기화
    # embedding_size = 1536
    # index = faiss.IndexFlatL2(embedding_size)
    # vectorstore = FAISS(embeddings, index, InMemoryDocstore({}), {})
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vector_store, decay_rate=0.0000000000000000000000001, k=2
    )

    # today = datetime.now()
    # yesterday = today - timedelta(days=1)

    # print("today:", today)
    # print("yesterday:", yesterday)
    docs = [
        Document(
            page_content="hello world yesterday",
            # metadata={"last_accessed_at": "1"},
        ),
        # Document(
        #     page_content="hello world today",
        #     # metadata={"last_accessed_at": "2"}
        # ),
    ]
    loader = TextLoader("./data/news2.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    retriever.add_documents(docs, ids=None, add_to_docstore=True)
    # "hello foo" 내용의 문서를 추가합니다.
    # retriever.add_documents(
    #     [
    #
    #     ]
    # )

    # Hello World는 가장 중요하기 때문에 먼저 반환되며, 감쇠율이 0에 가까워 여전히 최근이기 때문입니다.
    # docs = retriever.invoke("hello world")
    # print("====== low_rate retriever result ========")
    # for i, doc in enumerate(docs):
    #     print(doc.metadata)
    #     print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")


# def high_rate():
#     # 임베딩 모델 정의
#     embeddings_model = OpenAIEmbeddings()
#     # 벡터 스토어 초기화
#     embedding_size = 1536
#     index = faiss.IndexFlatL2(embedding_size)
#     vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})
#     retriever = TimeWeightedVectorStoreRetriever(
#         vectorstore=vectorstore, decay_rate=0.999, k=2
#     )
#
#     yesterday = datetime.now() - timedelta(days=1)
#     today = datetime.now()
#
#     retriever.add_documents(
#         [
#             Document(
#                 page_content="hello world yesterday",
#                 metadata={"last_accessed_at": yesterday},
#             )
#         ]
#     )
#     # "hello foo" 내용의 문서를 추가합니다.
#     retriever.add_documents(
#         [
#             Document(
#                 page_content="hello world today", metadata={"last_accessed_at": today}
#             )
#         ]
#     )
#
#     # Hello Foo가 먼저 반환됩니다. 왜냐하면 "hello world"가 대부분 잊혀졌기 때문입니다.
#     docs = retriever.invoke("hello world")
#     print("====== high_rate retriever result ========")
#     for i, doc in enumerate(docs):
#         print(doc.metadata)
#         print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")


# if __name__ == "__main__":
#     low_rate()
# high_rate()
