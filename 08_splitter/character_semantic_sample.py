from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

load_dotenv()


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"[doc {i+1}]" + d.page_content for i, d in enumerate(docs)]
        )
    )


# loader = TextLoader("data/semantic_sample.txt")
loader = TextLoader("data/sis_sample.txt")
document = loader.load()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def character_text_splitter():
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = text_splitter.split_documents(document)
    print("############# CharacterTextSplitter #############")
    pretty_print_docs(docs)


def recursive_character_text_splitter():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = text_splitter.split_documents(document)
    print("############# RecursiveCharacterTextSplitter #############")
    pretty_print_docs(docs)


def semantic_chunker():
    text_splitter = SemanticChunker(
        OpenAIEmbeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,
    )
    docs = text_splitter.create_documents([format_docs(document)])
    print("############# SemanticChunker #############")
    pretty_print_docs(docs)


# character_text_splitter()
# recursive_character_text_splitter()
semantic_chunker()
