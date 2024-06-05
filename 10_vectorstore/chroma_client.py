import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

# ChromaDB의 PersistentClient를 생성합니다.
persistent_client = chromadb.PersistentClient(path="./chroma_db")
# "my_chroma_collection"이라는 이름의 컬렉션을 가져오거나 생성합니다.
collection = persistent_client.get_or_create_collection(name="chroma_collection")
# 컬렉션에 ID와 문서를 추가합니다.
# collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])
# collection.add(ids=["1", "2"], documents=["a", "b"])
collection.add(
    documents=[
        "This is a document about pineapple",
    ],
    ids=["id2"],
)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#
#
# # Chroma 객체를 생성합니다.
langchain_chroma = Chroma(
    # PersistentClient를 전달합니다.
    client=persistent_client,
    # 사용할 컬렉션의 이름을 지정합니다.
    collection_name="chroma_collection",
    # 임베딩 함수를 전달합니다.
    embedding_function=embedding_function,
)

# 컬렉션의 항목 수를 출력합니다.
print(
    "현재 저장된 Collection 의 개수는 ",
    langchain_chroma._collection.count(),
    " 개 입니다.",
)

query = "pineapple"


loader = TextLoader("./data/news.txt")
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = loader.load_and_split(text_splitter)
# 간단한 ID 생성
ids = [str(i) for i in range(1, len(docs) + 1)]

# 데이터 추가
example_db = Chroma.from_documents(
    docs, embedding_function, ids=ids, persist_directory="./chroma_db"
)
docs = example_db.similarity_search(query)
print(docs[0].metadata)

# 문서의 메타데이터 업데이트
docs[0].metadata = {
    "source": "./images/appendix-keywords.txt",
    "new_value": "테스트용으로 업데이트할 내용입니다.",
}

# DB 에 업데이트
example_db.update_document(ids[0], docs[0])
print(example_db._collection.get(ids=[ids[0]]))

# 문서 개수 출력
print("count before", example_db._collection.count())
# 마지막 문서 삭제
example_db._collection.delete(ids=[ids[-1]])
# 삭제 후 문서 개수 출력
print("count after", example_db._collection.count())
