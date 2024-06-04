import time

from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

embedding = OpenAIEmbeddings()

store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embedding,
    store,
    namespace=embedding.model,
)

raw_documents = TextLoader("./data/past_love.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

start = time.time()
db = FAISS.from_documents(documents, cached_embedder)
end = time.time()
print(f"1번째: {end - start:.5f} sec")

start = time.time()
db2 = FAISS.from_documents(documents, cached_embedder)
end = time.time()
print(f"2번째: {end - start:.5f} sec")

result = list(store.yield_keys())[:5]
print(result)
