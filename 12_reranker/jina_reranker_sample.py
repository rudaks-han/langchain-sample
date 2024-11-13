from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from custom_jina_reranker import CustomJinaRerank
from sample_docs import (
    get_sample_docs,
    elapsed_time,
)

load_dotenv()

compressor = CustomJinaRerank(top_n=1)

embedding_model = OpenAIEmbeddings()
vector_store = Chroma.from_documents(get_sample_docs(), embedding_model)
# retriever = vector_store.as_retriever(search_kwargs={"k": 10})
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

query = "대한민국의 수도는?"


@elapsed_time
def execute():
    compressed_docs = compression_retriever.invoke(query)
    print(compressed_docs)


execute()
