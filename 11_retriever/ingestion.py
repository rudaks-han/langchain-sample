from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("./data/python.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    text = text_splitter.split_documents(document)
    print(f"created {len(text)} chunks")

    embeddings = OpenAIEmbeddings()

    Chroma.from_documents(
        text, embeddings, persist_directory="chroma_emb"
    )
