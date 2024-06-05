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

bm25_retriever = BM25Retriever.from_texts(
    doc_list_1, metadatas=[{"source": 1}] * len(doc_list_1)
)
bm25_retriever.k = 2

bm25_result = bm25_retriever.invoke(question)
print("[bm25_result]")
for doc in bm25_result:
    print(doc.page_content)


embedding = OpenAIEmbeddings()
faiss_vectorstore = FAISS.from_texts(
    doc_list_1, embedding, metadatas=[{"source": 2}] * len(doc_list_1)
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
faiss_results = faiss_retriever.invoke(question)

print("[faiss_results]")
for doc in faiss_results:
    print(doc.page_content)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

docs = ensemble_retriever.invoke(question)

print("[ensembleRetriever results]")
for doc in docs:
    print(doc.page_content)


# doc_list_1 = [
#     "비타민A : 당근, 시금치, 감자 등의 주황색과 녹색 채소에서 섭취할 수 있습니다.",
#     "비타민B : 전곡물, 콩, 견과류, 육류 등 다양한 식품에서 찾을 수 있습니다.",
#     "비타민C : 오렌지, 키위, 딸기, 브로콜리, 피망 등의 과일과 채소에 많이 들어 있습니다.",
#     "비타민D : 연어, 참치, 버섯, 우유, 계란 노른자 등에 함유되어 있습니다.",
#     "비타민E : 해바라기씨, 아몬드, 시금치, 아보카도 등에서 섭취할 수 있습니다.",
# ]
#
# # 비타민 별 효능 정보
# doc_list_2 = [
#     "비타민A : 시력과 피부 건강을 지원합니다.",
#     "비타민B : 에너지 대사와 신경계 기능을 돕습니다.",
#     "비타민C : 면역 체계를 강화하고 콜라겐 생성을 촉진합니다.",
#     "비타민D : 뼈 건강과 면역 체계를 지원합니다.",
#     "비타민E : 항산화 작용을 통해 세포를 보호합니다.",
# ]
#
# # bm25 retriever와 faiss retriever를 초기화합니다.
# bm25_retriever = BM25Retriever.from_texts(
#     # doc_list_1의 텍스트와 메타데이터를 사용하여 BM25Retriever를 초기화합니다.
#     doc_list_1,
#     metadatas=[{"source": 1}] * len(doc_list_1),
# )
# bm25_retriever.k = 2
#
# vectorstore = Chroma(collection_name="ensemble", embedding_function=OpenAIEmbeddings())
#
# query = "비타민A 의 효능은?"
# retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
#
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, retriever],
#     weights=[0.6, 0.4],
#     search_type="mmr",
# )
#
# # 검색 결과 문서를 가져옵니다.
#
# ensemble_result = ensemble_retriever.invoke(query)
# bm25_result = bm25_retriever.invoke(query)
# vector_result = retriever.invoke(query)
#
# # 가져온 문서를 출력합니다.
# print("[Ensemble Retriever]\n", ensemble_result, end="\n\n")
# print("[BM25 Retriever]\n", bm25_result, end="\n\n")
# print("[FAISS Retriever]\n", vector_result, end="\n\n")
