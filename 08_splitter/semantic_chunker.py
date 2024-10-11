from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# loader = TextLoader("data/sis_sample.txt")
# loader = TextLoader("data/sis_sample_1.txt")
loader = TextLoader("data/semantic_chunk_sample.txt")
document = loader.load()

embedding = OpenAIEmbeddings()

store = LocalFileStore("./cache/")

embedding = CacheBackedEmbeddings.from_bytes_store(
    embedding,
    store,
    namespace=embedding.model,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"[doc {i+1}]" + d.page_content for i, d in enumerate(docs)]
        )
    )


def html_style():
    return "<style>table {width: 100%;}table, th, td {border: 1px solid black;border-collapse: collapse;}th, td {padding: 15px;text-align: left;}table tr:nth-child(even) {background-color: #f2f2f2;}table tr:nth-child(odd) {background-color: #ffffff;}</style>"


def write_to_html(docs, filename):
    with open(f"data/{filename}", "w") as f:
        f.write(
            f"<html>{html_style()}<body><h1>{filename}</h1><table border=1>{([f'<tr><td>[doc {i+1}] {d.page_content}</td></tr>' for i, d in enumerate(docs)])}</table></body></html>"
        )


def semantic_chunker_percentile():
    text_splitter = SemanticChunker(
        embedding,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    docs = text_splitter.create_documents([format_docs(document)])
    print("############# SemanticChunker percentile #############")
    pretty_print_docs(docs)
    # write_to_html(docs, "percentile.html")


def semantic_chunker_standard_deviation():
    text_splitter = SemanticChunker(
        embedding, breakpoint_threshold_type="standard_deviation"
    )
    docs = text_splitter.create_documents([format_docs(document)])
    print("############# SemanticChunker standard_deviation #############")
    # pretty_print_docs(docs)
    write_to_html(docs, "standard_deviation.html")


def semantic_chunker_standard_interquartile():
    text_splitter = SemanticChunker(
        embedding, breakpoint_threshold_type="interquartile"
    )
    docs = text_splitter.create_documents([format_docs(document)])
    print("############# SemanticChunker standard_deviation #############")
    # pretty_print_docs(docs)
    write_to_html(docs, "interquartile.html")


semantic_chunker_percentile()
# semantic_chunker_standard_deviation()
# semantic_chunker_standard_interquartile()
