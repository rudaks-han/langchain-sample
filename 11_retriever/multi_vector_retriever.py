import uuid

import langchain
from dotenv import load_dotenv
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

langchain.debug = False

load_dotenv()

loaders = [
    TextLoader("./data/news2.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
# 기본 문서도 너무 크면 안되므로 4000자로 짜른다.
split_docs = text_splitter.split_documents(docs)


def small_chunk():
    # child 청크를 저장할 vectorstore를 생성
    vectorstore = Chroma(
        persist_directory="chroma_multi_vector_store",
        collection_name="full_documents",
        embedding_function=OpenAIEmbeddings(),
    )

    # parent 문서 저장 레이어
    store = InMemoryByteStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )

    doc_ids = [str(uuid.uuid4()) for _ in split_docs]

    # child 청크를 만들기 위한 splitter
    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    sub_docs = []
    for i, doc in enumerate(split_docs):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)

    retriever.vectorstore.add_documents(sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, split_docs)))

    question = "비타민"
    # vectorstore는 작은 청크를 조회
    docs = retriever.vectorstore.similarity_search(question)
    print("====== child docs result ========")
    for i, doc in enumerate(docs):
        print(
            f"[문서 {i}][{len(doc.page_content)}] {doc.page_content.replace('\n', ' ')}"
        )

    full_docs = retriever.invoke(question)
    print("====== parent docs result ========")
    for i, doc in enumerate(full_docs):
        print(
            f"[문서 {i}][{len(doc.page_content)}] {doc.page_content.replace('\n', ' ')}"
        )


def summary():
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | ChatOpenAI(max_retries=0)
        | StrOutputParser()
    )

    summaries = chain.batch(split_docs, {"max_concurrency": 5})

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        persist_directory="chroma_multi_vector_store",
        collection_name="summaries",
        embedding_function=OpenAIEmbeddings(),
    )
    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"
    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )
    doc_ids = [str(uuid.uuid4()) for _ in split_docs]
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    question = "비타민"
    sub_docs = vectorstore.similarity_search(question)
    print("====== summary child docs result ========")
    for i, doc in enumerate(sub_docs):
        print(
            f"[문서 {i}][{len(doc.page_content)}] {doc.page_content.replace('\n', ' ')}"
        )

    retrieved_docs = retriever.invoke(question)
    print("====== summary docs result ========")
    for i, doc in enumerate(retrieved_docs):
        print(
            f"[문서 {i}][{len(doc.page_content)}] {doc.page_content.replace('\n', ' ')}"
        )


def hypothetical_queries():
    functions = [
        {
            "name": "hypothetical_questions",
            "description": "Generate hypothetical questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["questions"],
            },
        }
    ]

    chain = (
        {"doc": lambda x: x.page_content}
        # Only asking for 3 hypothetical questions, but this could be adjusted
        | ChatPromptTemplate.from_template(
            "Generate a list of exactly 3 hypothetical questions that the below document could be used to answer:\n\n{doc}"
        )
        | ChatOpenAI(max_retries=0, model="gpt-4").bind(
            functions=functions, function_call={"name": "hypothetical_questions"}
        )
        | JsonKeyOutputFunctionsParser(key_name="questions")
    )

    print("split_docs[0]", split_docs[0])
    hypothetical_docs = chain.invoke(split_docs[0])
    print("====== hypothetical docs result ========")
    for i, doc in enumerate(hypothetical_docs):
        print(f"[문서 {i}] {doc}")

    hypothetical_questions = chain.batch(split_docs, {"max_concurrency": 5})
    print("====== hypothetical_questions result ========")
    print(hypothetical_questions)

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        persist_directory="chroma_multi_vector_store",
        collection_name="hypo-questions",
        embedding_function=OpenAIEmbeddings(),
    )
    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"
    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )
    doc_ids = [str(uuid.uuid4()) for _ in docs]
    question_docs = []
    for i, question_list in enumerate(hypothetical_questions):
        question_docs.extend(
            [
                Document(page_content=s, metadata={id_key: doc_ids[i]})
                for s in question_list
            ]
        )

    retriever.vectorstore.add_documents(question_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    question = "비타민"
    sub_docs = vectorstore.similarity_search(question)
    print("====== hypothetical_questions result ========")
    for i, doc in enumerate(sub_docs):
        print(
            f"[문서 {i}][{len(doc.page_content)}] {doc.page_content.replace('\n', ' ')}"
        )

    retrieved_docs = retriever.invoke(question)
    print("====== hypothetical_retrieved docs result ========")
    for i, doc in enumerate(retrieved_docs):
        print(
            f"[문서 {i}][{len(doc.page_content)}] {doc.page_content.replace('\n', ' ')}"
        )


if __name__ == "__main__":
    small_chunk()
    # summary()
    # hypothetical_queries()
