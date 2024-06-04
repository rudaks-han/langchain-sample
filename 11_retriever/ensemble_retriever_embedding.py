import langchain
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

langchain.debug = True

load_dotenv()
# pip install -qU rank_bm25 deeplake

# 비타민 별 섭취할 수 있는 음식 정보
doc_list_1 = [
    "비타민A : 당근, 시금치, 감자 등의 주황색과 녹색 채소에서 섭취할 수 있습니다.",
    "비타민B : 전곡물, 콩, 견과류, 육류 등 다양한 식품에서 찾을 수 있습니다.",
    "비타민C : 오렌지, 키위, 딸기, 브로콜리, 피망 등의 과일과 채소에 많이 들어 있습니다.",
    "비타민D : 연어, 참치, 버섯, 우유, 계란 노른자 등에 함유되어 있습니다.",
    "비타민E : 해바라기씨, 아몬드, 시금치, 아보카도 등에서 섭취할 수 있습니다.",
]

# 비타민 별 효능 정보
doc_list_2 = [
    "비타민A : 시력과 피부 건강을 지원합니다.",
    "비타민B : 에너지 대사와 신경계 기능을 돕습니다.",
    "비타민C : 면역 체계를 강화하고 콜라겐 생성을 촉진합니다.",
    "비타민D : 뼈 건강과 면역 체계를 지원합니다.",
    "비타민E : 항산화 작용을 통해 세포를 보호합니다.",
]

# bm25 retriever와 faiss retriever를 초기화합니다.
bm25_retriever = BM25Retriever.from_texts(
    # doc_list_1의 텍스트와 메타데이터를 사용하여 BM25Retriever를 초기화합니다.
    doc_list_1,
    metadatas=[{"source": 1}] * len(doc_list_1),
)
bm25_retriever.k = 1  # BM25Retriever의 검색 결과 개수를 1로 설정합니다.

vectorstore = Chroma(collection_name="ensemble", embedding_function=OpenAIEmbeddings())

query = "비타민A 의 효능은?"
# 벡터 저장소를 사용하여 retriever를 생성하고, 검색 결과 개수를 1로 설정합니다.
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 앙상블 retriever를 초기화합니다.
ensemble_retriever = EnsembleRetriever(
    # BM25Retriever와 FAISS retriever를 사용하여 EnsembleRetriever를 초기화하고, 각 retriever의 가중치를 0.6:0.4로 설정합니다.
    retrievers=[bm25_retriever, retriever],
    weights=[0.6, 0.4],
    search_type="mmr",
)

# 검색 결과 문서를 가져옵니다.

ensemble_result = ensemble_retriever.invoke(query)
bm25_result = bm25_retriever.invoke(query)
vector_result = retriever.invoke(query)

# 가져온 문서를 출력합니다.
print("[Ensemble Retriever]\n", ensemble_result, end="\n\n")
print("[BM25 Retriever]\n", bm25_result, end="\n\n")
print("[FAISS Retriever]\n", vector_result, end="\n\n")
