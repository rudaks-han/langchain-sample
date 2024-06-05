from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter


loader = TextLoader("./data/news.txt")
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = loader.load_and_split(text_splitter)
# 간단한 ID 생성
ids = [str(i) for i in range(1, len(docs) + 1)]

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
example_db = Chroma.from_documents(
    docs, embedding_function, ids=ids, persist_directory="./chroma_db"
)
query = "문화체육관광부"
docs = example_db.similarity_search(query)
print(f"docs[0].metadata: {docs[0].metadata}")

docs[0].metadata = {
    "source": "./images/appendix-keywords.txt",
    "new_value": "테스트용으로 업데이트할 내용입니다.",
}

# DB 에 업데이트
print(f"ids[0] update metadata : {docs[0].metadata}")
example_db.update_document(ids[0], docs[0])
print(f"ids[0] : {example_db._collection.get(ids=[ids[0]])}")

# 문서 개수 출력
print("문서 개수", example_db._collection.count())
# 마지막 문서 삭제
example_db._collection.delete(ids=[ids[-1]])
# 삭제 후 문서 개수 출력
print("삭제 후 문서 개수", example_db._collection.count())
