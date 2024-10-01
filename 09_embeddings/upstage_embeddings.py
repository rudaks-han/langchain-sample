from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings

# API KEY 정보로드
load_dotenv()

embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
doc_result = embeddings.embed_documents(
    ["Sam is a teacher.", "This is another document"]
)
print(doc_result)
print(len(doc_result[0]))

# query_result = embeddings.embed_query("What does Sam do?")
# print(query_result)
