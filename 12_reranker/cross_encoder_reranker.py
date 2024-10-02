from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"[doc {i+1}]" + d.page_content for i, d in enumerate(docs)]
        )
    )


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader("./data/baseball.txt").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
retriever = Chroma.from_documents(texts, embeddings).as_retriever(
    search_kwargs={"k": 5}
)
query = "KBO 야구 룰에서 홈런을 많이 때린 팀이 이기는거야?"
docs = retriever.invoke(query)
print("###### retrieved docs ######")
pretty_print_docs(docs)


model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=model, top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
compressed_docs = compression_retriever.invoke(query)
print("###### compressed docs ######")
pretty_print_docs(compressed_docs)
