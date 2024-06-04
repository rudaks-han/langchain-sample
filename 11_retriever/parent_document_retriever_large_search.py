from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


parent_splitter = RecursiveCharacterTextSplitter(chunk_size=900)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=300)

vectorstore = Chroma(
    persist_directory="chroma_parent_doc",
    collection_name="split_parents",
    embedding_function=OpenAIEmbeddings(),
)

fs = LocalFileStore(root_path="./parent_document_store")
store = create_kv_docstore(fs)

# Retriever 를 생성합니다.
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

question = "비타민은 우리몸에 어떤 역할을 하는가?"
sub_docs = vectorstore.similarity_search(question)

print(f"자식 문서에서의 결과 개수: {len(sub_docs)}")
print(sub_docs)

retrieved_docs = retriever.invoke(question)
print(f"부모 문서에서의 결과 개수: {len(retrieved_docs)}")
print(retrieved_docs)
print(len(retrieved_docs[0].page_content))
