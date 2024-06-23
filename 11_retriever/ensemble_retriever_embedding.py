from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

question = "갤럭시 S21의 특징은?"

doc_list_1 = [
    "Galaxy S9의 특징은 저렴하다는 것이다",
    "Galaxy S9의 배터리는 3000 mAh이다",
    "Galaxy S10의 카메라는 Triple rear cameras이다. ",
    "Galaxy S20의 Display는 6.2-inch Dynamic AMOLED이다.",
    "Galaxy S20의 저장공간은 128G이다",
    "Galaxy S21의 Ram은 8GB이다",
]


embedding = OpenAIEmbeddings()
faiss_vectorstore = FAISS.from_texts(
    doc_list_1, embedding, metadatas=[{"source": 2}] * len(doc_list_1)
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
faiss_results = faiss_retriever.invoke(question)

print(f"####################### vector 검색 결과 #################")
for doc in faiss_results:
    print(doc.page_content)

bm25_retriever = BM25Retriever.from_texts(
    doc_list_1, metadatas=[{"source": 1}] * len(doc_list_1)
)
bm25_retriever.k = 2

bm25_result = bm25_retriever.invoke(question)
print(f"####################### bm25 검색 결과 #################")
for doc in bm25_result:
    print(doc.page_content)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

docs = ensemble_retriever.invoke(question)

print(f"####################### ensemble 검색 결과 #################")
for doc in docs:
    print(doc.page_content)
