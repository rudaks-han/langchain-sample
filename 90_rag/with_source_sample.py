from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

embeddings = OpenAIEmbeddings()
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = vector_store.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | model
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


result = rag_chain_with_source.invoke("한모 씨의 여름 휴가는 언제야?")
print(result)
