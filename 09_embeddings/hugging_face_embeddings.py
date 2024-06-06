# pipenv install langchain sentence_transformers

import os

from dotenv import load_dotenv
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
)

load_dotenv()


inference_api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
os.environ["HF_HOME"] = "./cache/"

embeddings = HuggingFaceEmbeddings()
# embeddings = HuggingFaceBgeEmbeddings()

text = "동해물과 백두산이 마르고 닳도록"

# 텍스트를 임베딩하여 쿼리 결과를 생성합니다.
query_result = embeddings.embed_query(text)
print(query_result[:3])
