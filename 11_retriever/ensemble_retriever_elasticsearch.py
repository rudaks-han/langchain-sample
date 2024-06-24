import os

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

from custom_elastic_search_store import CustomElasticSearchStore

load_dotenv()

question = "갤럭시 S21의 특징은?"

text_list = [
    "Galaxy S9의 특징은 저렴하다는 것이다",
    "Galaxy S9의 배터리는 3000 mAh이다",
    "Galaxy S10의 카메라는 Triple rear cameras이다. ",
    "Galaxy S20의 Display는 6.2-inch Dynamic AMOLED이다.",
    "Galaxy S20의 저장공간은 128G이다",
    "Galaxy S21의 Ram은 8GB이다",
]

index_name = "test_docs"
embeddings = OpenAIEmbeddings()
vector_store = CustomElasticSearchStore(
    embedding=embeddings,
    index_name=index_name,
    es_url=os.getenv("ELASTICSEARCH_URL"),
)

# embedding
# for text in text_list:
#     doc = [Document(page_content=text)]
#     vector_store.add_documents(doc, add_to_docstore=True)

elasticsearch_client = vector_store.client

# 기본 bm25 검색
# bm25_retriever = ElasticSearchBM25Retriever(
#     client=elasticsearch_client, index_name=index_name
# )
# bm25_docs = bm25_retriever.invoke(question)
#
# print(f"####################### bm25 검색 결과 #################")
# for doc in bm25_docs:
#     print(doc.page_content)


# custom bm25 검색

# bm25_retriever = CustomElasticSearchBM25Retriever(
#     client=elasticsearch_client, index_name=index_name, search_args={"k": 3}
# )
# bm25_docs = bm25_retriever.invoke(question)
#
# print(f"####################### custom bm25 검색 결과 #################")
# for doc in bm25_docs:
#     print(doc.page_content)

# vector store 검색
search_kwargs = {"k": 1, "filter": {}}
# search_kwargs["filter"] = {"site": "naver"}
retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
vector_result = retriever.invoke(question)
print("====== custom vector store result ========")
for i, doc in enumerate(vector_result):
    print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")
