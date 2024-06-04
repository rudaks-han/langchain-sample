from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

load_dotenv()

embeddings = OpenAIEmbeddings()

loader = TextLoader("./data/news2.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 자식 분할기를 생성합니다.
child_splitter = RecursiveCharacterTextSplitter(chunk_size=300)

# DB를 생성합니다.
vectorstore = Chroma(
    persist_directory="chroma_parent_doc",
    collection_name="full_documents",
    embedding_function=OpenAIEmbeddings(),
)

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

retriever.add_documents(docs, ids=None, add_to_docstore=True)
sub_docs = vectorstore.similarity_search("비타민은 우리몸에 어떤 역할을 하는가?")
print(sub_docs)
# print(sub_docs[0].page_content)


def test():
    vector_store = Chroma(
        persist_directory="chroma_news", embedding_function=embeddings
    )

    loader = TextLoader("./data/news2.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    Chroma.from_documents(texts, embeddings, persist_directory="chroma_news")

    question = "조선 건국은 언제이며 대한제국에서 조선 왕조를 이어가는 시기는 언제인가?"

    results = vector_store.similarity_search_with_score(question)
    for result in results:
        print("\n")
        print(result[1])
        print(result[0].page_content)


# loaders = [
#     # 파일을 로드합니다.
#     # 파일을 로드합니다.
#     TextLoader("./data/보험약관.txt"),
# ]
# docs = []  # 빈 리스트를 생성합니다.
# for loader in loaders:  # loaders 리스트의 각 로더에 대해 반복합니다.
#     docs.extend(
#         loader.load()
#     )  # 로더를 사용하여 문서를 로드하고 docs 리스트에 추가합니다.
#
# # 자식 분할기를 생성합니다.
# child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
#
# # DB를 생성합니다.
# vectorstore = Chroma(
#     collection_name="full_documents", embedding_function=OpenAIEmbeddings()
# )
#
# store = InMemoryStore()
#
# # Retriever 를 생성합니다.
# retriever = ParentDocumentRetriever(
#     vectorstore=vectorstore,
#     docstore=store,
#     child_splitter=child_splitter,
# )
#
# # 문서를 검색기에 추가합니다. docs는 문서 목록이고, ids는 문서의 고유 식별자 목록입니다.
# retriever.add_documents(docs, ids=None, add_to_docstore=True)
#
# # 저장소의 모든 키를 리스트로 반환합니다.
# list(store.yield_keys())
#
# # 유사도 검색을 수행합니다.
# question = "사람이 죽는 경우 보상이 돼?"
# sub_docs = vectorstore.similarity_search(question)
#
# # sub_docs 리스트의 첫 번째 요소의 page_content 속성을 출력합니다.
# print(f"결과 개수: {len(sub_docs)}")
# print(sub_docs)
#
# # for sub_doc in sub_docs:
# #     print("______")
# #     print(sub_doc.page_content)
#
# retrieved_docs = retriever.invoke(question)
# print(f"retrieved_docs 결과 개수: {len(retrieved_docs)}")
# print(retrieved_docs)
