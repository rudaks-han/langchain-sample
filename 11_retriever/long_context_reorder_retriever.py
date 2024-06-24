from operator import itemgetter

import langchain
from langchain.prompts import PromptTemplate
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import format_document, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

langchain.debug = True

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="{page_content} [source:출처]"
)


embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()
question = "테슬라와 일론머스크에 대해 알려줄 수 있어?"

texts = [
    "기계 학습은 많은 산업을 혁신하고 있습니다.",
    "모나리자는 세계에서 가장 유명한 그림 중 하나입니다.",
    "인공지능은 더 나은 진단을 위해 의료 분야에 적용될 수 있습니다.",
    "이 문서는 자동차 산업에 대한 AI의 영향을 다룹니다.",
    "나는 공상 과학 소설을 읽는 것을 좋아합니다.",
    "테슬라의 자율 주행 기술은 자동차 분야의 선도적인 혁신입니다.",
    "이것은 특정한 문맥이 없는 일반적인 문장입니다.",
    "신경망은 현대 AI 시스템의 핵심 구성 요소입니다.",
    "AI 윤리는 오늘날 기술 중심의 세계에서 중요한 주제입니다.",
    "일론 머스크의 사업들은 기술 분야에 큰 영향을 미쳤습니다.",
]

retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)

template = """Given this text extracts:
    {context}
    
    -----
    Please answer the following question:
    {question}
    
    Answer in the following languages: {language}
    """


def run():

    docs = run_retriever(question)
    run_long_context_reorder(docs)
    # run_reorder_document(docs)
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {
            "context": itemgetter("question")
            | retriever
            | RunnableLambda(reorder_documents),
            "question": itemgetter("question"),
            "language": itemgetter("language"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({"question": question, "language": "KOREAN"})
    print("====== 답변 검색 result ========")
    print(answer)


def run_retriever(question):
    docs = retriever.invoke(question)
    print("====== 기본 검색 result ========")
    for i, doc in enumerate(docs):
        print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")

    return docs


def run_long_context_reorder(docs):
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)

    print("====== reordering result ========")
    for i, doc in enumerate(reordered_docs):
        print(f"[문서 {i}] {doc.page_content.replace('\n', ' ')}")

    return reordered_docs


def run_reorder_document(docs):
    reordered_docs_string = reorder_documents(docs)
    print(f"####################### reordering string 검색 결과 #################")
    print(reordered_docs_string)


def combine_documents(
    docs,
    document_prompt=DEFAULT_DOCUMENT_PROMPT,
    document_separator="\n",
):
    doc_strings = [
        f"[{i}] {format_document(doc, document_prompt)}" for i, doc in enumerate(docs)
    ]
    return document_separator.join(doc_strings)


def reorder_documents(docs):
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    combined = combine_documents(reordered_docs, document_separator="\n")
    print("====== combined result ========")
    print(combined)
    return combined


if __name__ == "__main__":
    run()
