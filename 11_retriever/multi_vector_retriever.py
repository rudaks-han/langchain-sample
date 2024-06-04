import uuid

from dotenv import load_dotenv
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loaders = [
    # TextLoader("./data/news.txt"),
    TextLoader("./data/news2.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    persist_directory="chroma_multi_vector_store",
    collection_name="full_documents",
    embedding_function=OpenAIEmbeddings(),
)
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

doc_ids = [str(uuid.uuid4()) for _ in docs]

parent_text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

parent_docs = []

for i, doc in enumerate(docs):
    _id = doc_ids[i]  # 현재 문서의 ID를 가져옵니다.
    # 현재 문서를 하위 문서로 분할합니다.
    parent_doc = parent_text_splitter.split_documents([doc])
    for _doc in parent_doc:  # 분할된 문서에 대해 반복합니다.
        # 문서의 메타데이터에 ID를 저장합니다.
        _doc.metadata[id_key] = _id
    parent_docs.extend(parent_doc)  # 분할된 문서를 리스트에 추가합니다.

child_docs = []  # 하위 문서를 저장할 리스트를 초기화합니다.
for i, doc in enumerate(docs):
    _id = doc_ids[i]  # 현재 문서의 ID를 가져옵니다.
    # 현재 문서를 하위 문서로 분할합니다.
    child_doc = child_text_splitter.split_documents([doc])
    for _doc in child_doc:  # 분할된 하위 문서에 대해 반복합니다.
        # 하위 문서의 메타데이터에 ID를 저장합니다.
        _doc.metadata[id_key] = _id
    child_docs.extend(child_doc)  # 분할된 하위 문서를 리스트에 추가합니다.

print(f"분할된 parent_docs의 개수: {len(parent_docs)}")
print(f"분할된 child_docs의 개수: {len(child_docs)}")

# 벡터 저장소에 하위 문서를 추가합니다.
retriever.vectorstore.add_documents(parent_docs)
retriever.vectorstore.add_documents(child_docs)

# 문서 저장소에 문서 ID와 문서를 매핑하여 저장합니다.
retriever.docstore.mset(list(zip(doc_ids, docs)))


# Vectorstore alone retrieves the small chunks
retriever.vectorstore.similarity_search("비타민")[0]

# Retriever returns larger chunks
result = retriever.invoke("비타민")
print(result)
# len(retriever.invoke("비타민")[0].page_content)

# from langchain.retrievers.multi_vector import SearchType
#
# retriever.search_type = SearchType.mmr
#
# len(retriever.invoke("비타민")[0].page_content)
