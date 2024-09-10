from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# filename = "data/news.txt"
# filename = "data/past_love.txt"
# filename = "data/sample.txt"
filename = "data/baseball.txt"
loader = TextLoader(filename)
document = loader.load()


def character_text_splitter(document):
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = text_splitter.split_documents(document)
    print_docs(docs)


def recursive_character_text_splitter(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=10)
    docs = text_splitter.split_documents(document)
    print_docs(docs)


def print_docs(documents):
    for i, doc in enumerate(documents):
        print(f"--------------- len: {len(doc.page_content)} ------------")
        print(f"{doc.page_content}")


if __name__ == "__main__":
    # character_text_splitter(document)
    recursive_character_text_splitter(document)
