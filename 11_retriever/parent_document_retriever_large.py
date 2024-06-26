import langchain
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

langchain.debug = True

load_dotenv()

loaders = [
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

# store = InMemoryStore()
store = LocalFileStore(root_path="./local_file_store")
fs = LocalFileStore(root_path="./file_store")
store = create_kv_docstore(fs)

# Retriever 를 생성합니다.
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs)

# list(store.yield_keys())

question = "비타민은 우리몸에 어떤 역할을 하는가?"
sub_docs = vectorstore.similarity_search(question)

print(f"결과 개수: {len(sub_docs)}")
print(sub_docs)

# for sub_doc in sub_docs:
#     print("______")
#     print(sub_doc.page_content)

retrieved_docs = retriever.invoke(question)
print(f"retrieved_docs 결과 개수: {len(retrieved_docs)}")
print(retrieved_docs)
