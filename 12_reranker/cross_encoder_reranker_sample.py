from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import OpenAIEmbeddings

from sample_docs import get_sample_docs, pretty_print_docs, elapsed_time

load_dotenv()


embedding_model = OpenAIEmbeddings()
vector_store = Chroma.from_documents(get_sample_docs(), embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

query = "대한민국의 수도는?"

retrieved_docs = retriever.get_relevant_documents(query)
print("###### retrieved docs ######")
pretty_print_docs(retrieved_docs)


@elapsed_time
def cross_encoder_reranker():
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    compressed_docs = compression_retriever.invoke(query)
    print("###### compressed docs ######")
    pretty_print_docs(compressed_docs)


cross_encoder_reranker()
