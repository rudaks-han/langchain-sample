from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loaders = [
    TextLoader("./data/news.txt"),
    TextLoader("./data/news2.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=900)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=300)

vectorstore = Chroma(
    persist_directory="chroma_parent_doc",
    collection_name="split_parents",
    embedding_function=OpenAIEmbeddings(),
)

fs = LocalFileStore(root_path="./parent_document_store")
store = create_kv_docstore(fs)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs, ids=None, add_to_docstore=True)
