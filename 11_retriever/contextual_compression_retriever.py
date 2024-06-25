import langchain
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

langchain.debug = True

load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()
vector_store = Chroma(persist_directory="chroma_emb", embedding_function=embeddings)

question = "박병호는 인천에서 몇 개의 홈런을 쳤나?"
retriever = vector_store.as_retriever()


# docs = retriever.invoke(question)
# print(f"####################### 기본 검색 결과 #################")
# for i, doc in enumerate(docs):
#     # print(doc.metadata)
#     print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")

# 기본 (LLM 호출)
# compression_retriever = chain_extractor_retriever()
# compressor = LLMChainExtractor.from_llm(llm)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=retriever,
# )
# compressed_docs = compression_retriever.invoke(question)
# print(f"####################### 압축 검색 결과 #################")
# for i, doc in enumerate(compressed_docs):
#     print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")

# LLMChainFilter (LLM 호출)
# _filter = LLMChainFilter.from_llm(llm)
#
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=_filter,
#     base_retriever=retriever,
# )
# compressed_docs = compression_retriever.invoke(question)
# print(f"####################### LLMChainFilter 압축 검색 결과 #################")
# for i, doc in enumerate(compressed_docs):
#     # print(doc.metadata)
#     print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")

# EmbeddingsFilter (LLM 호출 안함)
# embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.86)
#
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=embeddings_filter,
#     base_retriever=retriever,
# )
# #
# compressed_docs = compression_retriever.invoke(question)
# print(f"####################### EmbeddingsFilter 압축 검색 결과 #################")
# for i, doc in enumerate(compressed_docs):
#     print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")

# Pipeline (LLM 호출 안함)
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.86)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=retriever,
)

compressed_docs = compression_retriever.invoke(question)
print(f"####################### 압축 검색 결과 #################")
for i, doc in enumerate(compressed_docs):
    print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")

print("_________________ end ___________________")
