# pipenv install langchain sentence_transformers

from dotenv import load_dotenv

load_dotenv()

import os

# 사용자로부터 HuggingFace Inference API 키를 입력받습니다.
inference_api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

import os

# ./cache/ 경로에 다운로드 받도록 설정
os.environ["HF_HOME"] = "./cache/"

from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceBgeEmbeddings,
)

embeddings = HuggingFaceEmbeddings()  # HuggingFace 임베딩을 생성합니다.
embeddings = HuggingFaceBgeEmbeddings()

text = (
    "임베딩 테스트를 하기 위한 샘플 문장입니다."  # 테스트용 문서 텍스트를 정의합니다.
)

# 텍스트를 임베딩하여 쿼리 결과를 생성합니다.
query_result = embeddings.embed_query(text)

print(query_result[:3])

doc_result = embeddings.embed_documents(
    [text]
)  # 텍스트를 임베딩하여 문서 벡터를 생성합니다.
