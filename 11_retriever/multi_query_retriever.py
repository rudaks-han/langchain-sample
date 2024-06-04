import logging

import langchain
from dotenv import load_dotenv
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.multi_query import LineListOutputParser
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Load blog post
loader = WebBaseLoader("https://ko.wikipedia.org/wiki/%EB%B9%84%ED%83%80%EB%AF%BC")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# VectorDB
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

question = "비타민은 무엇인가?"
llm = ChatOpenAI(temperature=0)
retriever = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(), llm=llm)

langchain.debug = True
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

docs = retriever.invoke(question)
print(len(docs))
print(docs)

from langchain_core.prompts import PromptTemplate

load_dotenv()

output_parser = LineListOutputParser()

prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
llm = ChatOpenAI(temperature=0)

# Chain
chain = {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
# Other inputs
question = "비타민은 무엇인가?"

# Run
multiquery_retriever = MultiQueryRetriever.from_llm(
    llm=chain, retriever=vectordb.as_retriever()
)

# Results
unique_docs = retriever.invoke(question)
len(unique_docs)


def multiQueryRetriever():
    # Load blog post
    loader = WebBaseLoader("https://ko.wikipedia.org/wiki/%EB%B9%84%ED%83%80%EB%AF%BC")
    data = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(data)

    # VectorDB
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

    question = "비타민은 무엇인가?"
    llm = ChatOpenAI(temperature=0)
    retriever = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(), llm=llm)

    langchain.debug = True
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    docs = retriever.invoke(question)
    print(len(docs))
    print(docs)


#
# load_dotenv()
#
#
# if __name__ == "__main__":
#     print("Retrieving...")
#     chat = ChatOpenAI()
#     embeddings = OpenAIEmbeddings()
#
#     db = Chroma(persist_directory="chroma_emb", embedding_function=embeddings)
#
#     llm = ChatOpenAI(temperature=0)
#
#     retriever = db.as_retriever()
#     docs = retriever.invoke("Python이란 무엇인가?")
#     print(docs)
#
#     print(f"as_retriever 검색 수: {len(docs)}")
#     multiquery_retriever = MultiQueryRetriever.from_llm(  # MultiQueryRetriever를 언어 모델을 사용하여 초기화합니다.
#         retriever=db.as_retriever(),
#         llm=llm,
#     )
#
#     result = multiquery_retriever.invoke("Python은 무엇인가?")
#     print(result)
