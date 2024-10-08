from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 로드, 게시물 내용을 청크, 인덱싱한다.
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/mnews/article/029/0002905111",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("media_end_head_headline", "newsct_article _article_body")
        )
    ),
)
docs = loader.load()
print(docs)
# 문서 길이: 3780
len(docs[0].page_content)
# print(docs[0].page_content[:500])


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
splits = text_splitter.split_documents(docs)
print(f"splits: {len(splits)}")

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
retrieved_docs = retriever.invoke("한국을 뜨고 싶은 이유가 뭐야?")

for i, doc in enumerate(retrieved_docs):
    print(f"[{i}]: {doc.page_content}")


# print(len(retrieved_docs))
# print(retrieved_docs[0].page_content)

# prompt = hub.pull("rlm/rag-prompt")
# prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
#
# Question: {question}
#
# Context: {context}
#
# Answer:
# """
# prompt = PromptTemplate.from_template(prompt_template)
#
#
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)
#
#
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
#
# response = rag_chain.invoke("한국을 뜨고 싶은 이유가 뭐야?")
# print(response)
