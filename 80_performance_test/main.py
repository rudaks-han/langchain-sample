import random
import time
from functools import wraps

import uvicorn
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from fastapi import FastAPI
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.retrievers import ElasticSearchBM25Retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI, AsyncOpenAI

load_dotenv()


def elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print("Sync execution time:", (end_time - start_time))
        return result

    return wrapper


def elapsed_time_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        print("Async execution time:", (end_time - start_time))
        return result

    return wrapper


app = FastAPI()


@app.get("/request/sync")
def request():
    return "ok"


@app.get("/request/async")
async def request():
    return "ok"


template = ChatPromptTemplate.from_template("test template: {question}")
model = ChatOpenAI(temperature=0, model="gpt-4o-mini", base_url="http://localhost:7777")

client = OpenAI(api_key="xxx", base_url="http://localhost:7777")
client_async = AsyncOpenAI(api_key="xxx", base_url="http://localhost:7777")


@app.get("/request/openai/sync")
def openai():
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "한국의 수도는?"}],
        stream=False,
    )

    return completion.choices[0].message.content


@app.get("/request/openai/async")
async def openai():
    completion = await client_async.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "한국의 수도는?"}],
        stream=False,
    )

    return completion.choices[0].message.content


@app.get("/request/langchain/sync")
def request():
    chain = template | model
    result = chain.invoke({"question": "테스트 질문"})
    return result


@app.get("/request/langchain/async")
async def request():
    chain = template | model
    result = await chain.ainvoke({"question": "테스트 질문"})
    return result


embedding = OpenAIEmbeddings(
    base_url="http://localhost:7777/v1", model="text-embedding-3-small"
)

elasticsearch_url = "http://localhost:9200"
index_name = "index_name"
vector_store = ElasticsearchStore(
    es_url=elasticsearch_url,
    query_field="content",
    index_name=index_name,
    embedding=embedding,
    strategy=ElasticsearchStore.BM25RetrievalStrategy(),
    distance_strategy="COSINE",
)

es = Elasticsearch(elasticsearch_url)
vector_retriever = vector_store.as_retriever()
bm25_retriever = ElasticSearchBM25Retriever(
    client=es,
    index_name=index_name,
    k=3,
    search_filter={},
)
ensemble_retriever = EnsembleRetriever(
    # retrievers=[bm25_retriever, vector_retriever],
    # weights=[0.5, 0.5],
    # retrievers=[bm25_retriever],
    retrievers=[vector_retriever],
    # weights=[1.0],
)


search_kwargs = {"k": 1, "filter": {}}
question = "계좌 생성 방법"

vector_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)


@app.get("/request/retrieve/sync")
def request():
    vector_result = vector_retriever.invoke(question)
    return vector_result


@app.get("/request/retrieve/async")
async def request():
    vector_result = await vector_retriever.ainvoke(question)
    return vector_result


prompt = hub.pull("rlm/rag-prompt")
prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)


rag_chain = (
    {"context": vector_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


@app.get("/request/rag/sync")
def request():
    response = rag_chain.invoke("테스트 질문")
    return response


@app.get("/request/rag/async")
async def request():
    response = await rag_chain.ainvoke("테스트 질문")
    return response


template = """Answer the question based only on the following context:
{context}

Question: {input}
"""

prompt = ChatPromptTemplate.from_template(template)

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(vector_retriever, question_answer_chain)


@app.get("/request/retrieval_chain/sync")
def request():
    result = rag_chain.invoke(
        {
            "input": "한모 씨의 여름 휴가는 언제야?",
        }
    )
    return result


system_prompt = "Use the following pieces of context to answer the question at the end. Please follow the following rules:\n1. If the question isn't one you're looking for specific information on, please be kind.\n2. If context don't include info for the answer, don't try to make up an answer. Just say '{noResultMessage}'\n3. If you find the answer, write the answer in a concise way.\n4. Make sure to answer in Korean.\n\nContext:\n{context}"

contextualize_q_system_prompt = "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


@app.get("/request/rag/vector")
def request():
    session_id = f"thread-{random.randint(1, 100)}"
    # session_id = f"thread-01"
    # if store and session_id in store:
    #     store[session_id].clear()

    history_aware_retriever = create_history_aware_retriever(
        model, vector_retriever, contextualize_q_prompt
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    llm_result = conversational_rag_chain.invoke(
        {"input": "메시지...", "noResultMessage": "no message.."},
        config={"configurable": {"session_id": session_id}},
    )
    return llm_result


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    result = store[session_id]
    print(f"[{session_id}] message count: {len(result.messages)}")
    # print(f"___ {result.messages[-2:]}")
    # return store[session_id]
    # return result
    message_limit = 2
    if len(result.messages) > message_limit:
        return ChatMessageHistory(messages=result.messages[-message_limit:])
    else:
        return result
    # return ChatMessageHistory()


history_aware_retriever = create_history_aware_retriever(
    model, ensemble_retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


@app.get("/request/rag/ensemble")
def request():
    session_id = f"thread-{random.randint(1, 100)}"

    llm_result = conversational_rag_chain.invoke(
        {"input": "시스템", "noResultMessage": "no message.."},
        config={"configurable": {"session_id": session_id}},
    )
    return llm_result


if __name__ == "__main__":

    # 인자 추가
    uvicorn.run(
        "__main__:app",
        loop="uvloop",
    )
