import langchain
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
langchain.debug = True


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


embeddings = OpenAIEmbeddings()
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

retriever = vector_store.as_retriever()
template = """Answer the question based only on the following context:
{context}

current date: {current_date}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()

# LCEL을 사용하지 않는 방식
# retrieved_docs = retriever.invoke("한모 씨의 여름 휴가는 언제야?")
# context = format_docs(retrieved_docs)
#
# chain = prompt | model | StrOutputParser()
#
# result = chain.invoke({"context": context, "question": "한모 씨의 여름 휴가는 언제야?"})
# print(result)


def current_date():
    return "2024-09-22"


# LCEL을 사용하는 방식
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | RunnablePassthrough.assign(current_date=lambda x: current_date())
    | prompt
    | model
    | StrOutputParser()
)

result = chain.invoke("한모 씨의 여름 휴가는 언제야?")
print(result)


# result = retrieval_chain.invoke("한모 씨의 여름 휴가는 언제야?")
