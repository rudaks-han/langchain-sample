from typing import Dict, Iterable

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_community.document_loaders import TextLoader
from langchain_core.embeddings import Embeddings
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# from langchain_community.document_loaders import TextLoader
# from langchain_community.vectorstores import ElasticKnnSearch

# from langchain_community.vectorstores import ElasticsearchStore

# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter

# from langchain_community.vectorstores import ElasticVectorSearch

load_dotenv()

embeddings = OpenAIEmbeddings()

es_url = "http://172.16.120.203:9200"
es_client = Elasticsearch(hosts=[es_url])
info = es_client.info()
print(info)

index_name = "test-langchain-retriever"
text_field = "text"
dense_vector_field = "fake_embedding"
num_characters_field = "num_characters"
texts = [
    "foo",
    "bar",
    "world",
    "hello world",
    "hello",
    "foo bar",
    "bla bla foo",
]


def create_index(
    es_client: Elasticsearch,
    index_name: str,
    text_field: str,
    dense_vector_field: str,
    num_characters_field: str,
):
    es_client.indices.create(
        index=index_name,
        mappings={
            "properties": {
                text_field: {"type": "text"},
                dense_vector_field: {"type": "dense_vector"},
                num_characters_field: {"type": "integer"},
            }
        },
    )


def index_data(
    es_client: Elasticsearch,
    index_name: str,
    text_field: str,
    dense_vector_field: str,
    embeddings: Embeddings,
    texts: Iterable[str],
    refresh: bool = True,
) -> None:
    create_index(
        es_client, index_name, text_field, dense_vector_field, num_characters_field
    )

    vectors = embeddings.embed_documents(list(texts))
    requests = [
        {
            "_op_type": "index",
            "_index": index_name,
            "_id": i,
            text_field: text,
            dense_vector_field: vector,
            num_characters_field: len(text),
        }
        for i, (text, vector) in enumerate(zip(texts, vectors))
    ]

    bulk(es_client, requests)

    if refresh:
        es_client.indices.refresh(index=index_name)

    return len(requests)


def vector_query(search_query: str) -> Dict:
    vector = embeddings.embed_query(search_query)  # same embeddings as for indexing
    return {
        "knn": {
            "field": dense_vector_field,
            "query_vector": vector,
            "k": 5,
            "num_candidates": 10,
        }
    }


loader = TextLoader("./data/news.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

db = ElasticsearchStore.from_documents(
    docs,
    embeddings,
    es_url="http://172.16.120.203:9200",
    index_name="test_index",
)

db.client.indices.refresh(index="test_index")

vector_retriever = ElasticsearchRetriever.from_es_params(
    index_name=index_name,
    body_func=vector_query,
    content_field=text_field,
    url=es_url,
)

vector_retriever.invoke("foo")

# loader = TextLoader("./data/news.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
#
# embeddings = OpenAIEmbeddings()
# index_name = "test_index"
#
# es_connection = Elasticsearch("http://localhost:9200")
# vectorstore = ElasticKnnSearch(
#     embedding=embeddings, index_name=index_name, es_connection=es_connection
# )
#
# db = ElasticsearchStore.from_documents(
#     docs,
#     embeddings,
#     es_url="http://172.16.120.203:9200",
#     index_name="test_index",
# )
#
# db.client.indices.refresh(index="test_index")
#
# query = "한국관광공사에서 진행하는 페스타에 대해 설명해줘"
#
# results = vectorstore.similarity_search(query)
#
# # results = db.similarity_search(query)
# print(results)

# embedding = OpenAIEmbeddings()
# vector_store = ElasticVectorSearch(
#     embedding=embedding,
#     elasticsearch_url="http://172.16.120.203:9200",
#     index_name="test_index",
# )

# vector_search = ElasticsearchStore(
#     es_url="http://172.16.120.203:9200", index_name="test_index", embedding=embedding
# )

# print(vector_store)
