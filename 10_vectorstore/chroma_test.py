# import
import os

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("./data/news.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

inference_api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
os.environ["HF_HOME"] = "./cache/"

embedding_function = HuggingFaceEmbeddings()

db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")

query = "한국관광공사가 진행하는 숙박세일페스타는 무엇인가?"
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
docs = db.similarity_search(query)
# docs = db.max_marginal_relevance_search(query)

for i, doc in enumerate(docs):
    print(f"{i} : {doc.page_content}")
