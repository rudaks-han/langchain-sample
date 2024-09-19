import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

embeddings = OpenAIEmbeddings()
persistent_client = chromadb.PersistentClient(path="./chroma_db")
vector_store = Chroma(
    client=persistent_client,
    embedding_function=embeddings,
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = vector_store.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {input}
"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain_from_docs = (
    {
        "input": lambda x: x["input"],
        # "question": lambda x: x["question"],
        "context": lambda x: format_docs(x["context"]),
    }
    | prompt
    | model
    | StrOutputParser()
)
retrieve_docs = (lambda x: x["input"]) | retriever

chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
    answer=rag_chain_from_docs
)

result = chain.invoke(
    {
        "input": "인터파크트리플의 쿠폰은 얼마까지 할인돼?",
        # "question": "인터파크트리플의 쿠폰은 얼마까지 할인돼?",
    }
)

print(result)
# {'input': '인터파크트리플의 쿠폰은 얼마까지 할인돼?', 'question': '인터파크트리플의 쿠폰은 얼마까지 할인돼?', 'context': [], 'answer': '인터파크트리플의 쿠폰은 최대 5,000원까지 할인됩니다.'}
