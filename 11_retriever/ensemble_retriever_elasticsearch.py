import os

from dotenv import load_dotenv
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai.embeddings import OpenAIEmbeddings

from custom_elastic_search_bm25 import CustomElasticSearchBM25Retriever

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
vector_store = ElasticsearchStore(
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

bm25_retriever = CustomElasticSearchBM25Retriever(
    client=elasticsearch_client, index_name=index_name, search_args={"k": 1}
)
bm25_docs = bm25_retriever.invoke(question)

print(f"####################### custom bm25 검색 결과 #################")
for doc in bm25_docs:
    print(doc.page_content)

# faiss_vectorstore = FAISS.from_texts(
#     doc_list_1, embedding, metadatas=[{"source": 2}] * len(doc_list_1)
# )
# faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
# faiss_results = faiss_retriever.invoke(question)
#
# print(f"####################### vector 검색 결과 #################")
# for doc in faiss_results:
#     print(doc.page_content)
#
# bm25_retriever = BM25Retriever.from_texts(
#     doc_list_1, metadatas=[{"source": 1}] * len(doc_list_1)
# )
# bm25_retriever.k = 2
#
# bm25_result = bm25_retriever.invoke(question)
# print(f"####################### bm25 검색 결과 #################")
# for doc in bm25_result:
#     print(doc.page_content)
#
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
# )
#
# docs = ensemble_retriever.invoke(question)
#
# print(f"####################### ensemble 검색 결과 #################")
# for doc in docs:
#     print(doc.page_content)
