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
        parse_only=bs4.SoupStrainer(class_=("media_end_head_headline", "newsct_body"))
    ),
)
docs = loader.load()
# 문서 길이: 3780
# len(docs[0].page_content)
# print(docs[0].page_content[:500])


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
splits = text_splitter.split_documents(docs)
# print(f"splits: {len(splits)}")

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

retrieved_docs = retriever.invoke("한국을 뜨고 싶은 이유가 뭐야?")

# print(len(retrieved_docs))
# print(retrieved_docs[0].page_content)


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


#
#
# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )
#
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )
#
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#
# response = rag_chain.invoke({"input": "한국을 뜨고 싶은 이유가 뭐야?"})
# print(response["answer"])
#
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#
#
# response = rag_chain.invoke({"input": "한국을 뜨고 싶은 이유가 뭐야?"})
#
# print(response)
# for document in response["context"]:
#     print("------------------")
#     print(document)


from langchain_core.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("한국을 뜨고 싶은 이유가 뭐야?")
print(result)
