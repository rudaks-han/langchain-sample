from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from sample_docs import (
    get_sample_docs,
    pretty_print_docs,
    elapsed_time,
    pretty_print_texts,
)

load_dotenv()


embedding_model = OpenAIEmbeddings()
vector_store = Chroma.from_documents(get_sample_docs(), embedding_model)
# retriever = vector_store.as_retriever(search_kwargs={"k": 10})
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

query = "대한민국의 수도는?"

retrieved_docs = retriever.invoke(query)
print("###### retrieved docs ######")
pretty_print_docs(retrieved_docs)


@elapsed_time
def flash_rank_rerank():
    # compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12", top_n=10)
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=10)
    # compressor = FlashrankRerank(top_n=10)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )
    compressed_docs = compression_retriever.invoke(query)

    print("###### flash_rank_rerank ######")
    pretty_print_docs(compressed_docs)


flash_rank_rerank()


@elapsed_time
def rerank_request():
    formatted_data = [{"text": doc.page_content} for doc in retrieved_docs]
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    rerankrequest = RerankRequest(query=query, passages=formatted_data)
    results = ranker.rerank(rerankrequest)
    print("###### RerankRequest ######")
    pretty_print_texts(results)


# rerank_request()
