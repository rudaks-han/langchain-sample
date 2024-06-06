from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import ElasticVectorSearch
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


def main():
    loader = TextLoader("./data/news.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = ElasticVectorSearch.from_documents(
        documents,
        embeddings,
        elasticsearch_url="http://172.16.120.203:9200",
        index_name="elastic-index",
    )
    print(vector_store.client.info())

    result = vector_store.similarity_search("한국관광공사")
    print(result)


if __name__ == "__main__":
    main()
