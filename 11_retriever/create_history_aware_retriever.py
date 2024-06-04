import langchain
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings

langchain.debug = True

load_dotenv()


if __name__ == "__main__":
    llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings()

    db = Chroma(persist_directory="chroma_emb", embedding_function=embeddings)
    retriever = db.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    from langchain_core.messages import HumanMessage, AIMessage

    chat_history = [
        HumanMessage(content="Can LangSmith help test my LLM applications?"),
        AIMessage(content="Yes!"),
    ]
    retriever_chain.invoke({"chat_history": chat_history, "input": "Tell me how"})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    chat_history = [
        HumanMessage(content="Can LangSmith help test my LLM applications?"),
        AIMessage(content="Yes!"),
    ]
    result = retrieval_chain.invoke(
        {"chat_history": chat_history, "input": "Tell me how"}
    )

    print(result)
