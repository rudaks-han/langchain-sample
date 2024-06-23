from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

embeddings = OpenAIEmbeddings()
vector_store = Chroma(persist_directory="chroma_emb", embedding_function=embeddings)

loader = TextLoader("./data/baseball.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
Chroma.from_documents(texts, embeddings, persist_directory="chroma_emb")

retriever = vector_store.as_retriever(search_type="mmr")
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8},
)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

question = "박병호는 인천에서 몇 개의 홈런을 쳤나?"


docs = vector_store.similarity_search_with_score(question)
print("====== similarity_search_with_score result ========")
for i, doc in enumerate(docs):
    print(f"[문서 {i}][{doc[1]}] {doc[0].page_content.replace('\n', ' ')}")


retriever = vector_store.as_retriever()
docs = retriever.invoke(question)

print("====== retriever result ========")
for i, doc in enumerate(docs):
    print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")


retriever = vector_store.as_retriever(search_type="mmr")
docs = retriever.invoke(question)

print("====== retriever mmr result ========")
for i, doc in enumerate(docs):
    print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")


retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8},
)

docs = retriever.invoke(question)

print("====== retriever score_threshold result ========")
for i, doc in enumerate(docs):
    print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")
