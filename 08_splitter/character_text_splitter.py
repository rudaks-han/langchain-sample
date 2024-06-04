from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


def split_text():
    loader = TextLoader("data/sample.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=20, chunk_overlap=10, length_function=len
    )
    texts = text_splitter.split_documents(document)

    for i, text in enumerate(texts):
        print(f"{i} : {text.page_content}")


def split_text2():
    loader = TextLoader("data/past_love.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=50, chunk_overlap=14, length_function=len
    )
    texts = text_splitter.split_documents(document)

    for i, text in enumerate(texts):
        print(f"=={i}==")
        print(f"{text.page_content}")


if __name__ == "__main__":
    # split_text()
    split_text2()
