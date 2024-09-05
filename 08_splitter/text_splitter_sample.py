from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# filename = "data/news.txt"
# filename = "data/past_love.txt"
filename = "data/sample.txt"
loader = TextLoader(filename)
document = loader.load()


def character_text_splitter(document):
    text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=10)
    texts = text_splitter.split_documents(document)

    for i, text in enumerate(texts):
        print(f"{i} : {text.page_content}")


def recursive_character_text_splitter(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=20, chunk_overlap=10, length_function=len
    )
    texts = text_splitter.split_documents(document)

    for i, text in enumerate(texts):
        print(f"{i} : {text.page_content}")


if __name__ == "__main__":
    character_text_splitter(document)
    # recursive_character_text_splitter(document)
