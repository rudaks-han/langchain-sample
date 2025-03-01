from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings()
# vector_store = Chroma(persist_directory="chroma_emb", embedding_function=embeddings)


vector_store = ElasticsearchStore(
    es_url="http://172.16.110.139:9200",
    es_user="elastic",
    es_password="tmvprxmfk00",
    query_field="content",
    index_name="attic_aiadapter_bot_data",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    # strategy=ElasticsearchStore.ExactRetrievalStrategy(),
    strategy=ElasticsearchStore.ApproxRetrievalStrategy(),
    distance_strategy="COSINE",
)

retriever = vector_store.as_retriever(
    # search_type="similarity_score_threshold",
    search_type="mmr",
    search_kwargs={"score_threshold": 0.8},
)

question = "계좌 생성 방법"
docs = retriever.invoke(question)
#
print("====== retriever score_threshold result ========")
for i, doc in enumerate(docs):
    print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")
