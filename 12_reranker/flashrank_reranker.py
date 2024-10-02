from dotenv import load_dotenv
from langchain_chroma import Chroma
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

# 문서 로드
documents = TextLoader("./data/baseball.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
#
retriever = Chroma.from_documents(texts, embeddings).as_retriever(
    search_kwargs={"k": 5}
)
query = "KBO 야구 룰에서 홈런을 많이 때린 팀이 이기는거야?"
#
docs = retriever.invoke(query)
print("###### retrieved docs ######")
pretty_print_docs(docs)


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
formatted_data = [{"text": doc.page_content} for doc in docs]
compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
compressed_docs = compression_retriever.invoke(query)

print("###### compressed docs ######")
pretty_print_docs(compressed_docs)


# ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
# rerankrequest = RerankRequest(query=query, passages=formatted_data)
# results = ranker.rerank(rerankrequest)
# print(results)
# print(
#     f"\n{'-' * 100}\n".join([f"[doc {i+1}]" + d["text"] for i, d in enumerate(results)])
# )
