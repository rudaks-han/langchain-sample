import bs4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

# embeddings = OpenAIEmbeddings()
#
# loader = TextLoader("./data/news.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
#
# # 데이터 추가
# Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")


# 1. Load, chunk and index the contents of the blog to create a retriever.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    persist_directory="./chroma_db", documents=splits, embedding=OpenAIEmbeddings()
)
