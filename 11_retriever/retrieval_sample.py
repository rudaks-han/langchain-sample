import langchain
from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

langchain.debug = True

load_dotenv()


if __name__ == "__main__":
    print("Retrieving...")
    llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings()

    db = Chroma(persist_directory="chroma_emb", embedding_function=embeddings)

    retriever = db.as_retriever()
    # chain = RetrievalQA.from_chain_type(
    #     llm=llm, retriever=retriever, chain_type="stuff"
    # )

    prompt_text = """
            Use the following pieces of context to answer the question at the end. Please follow the following rules:
            2. If you don't know the answer, don't try to make up an answer. Just say **{noResultMessage}** and add the source links as a list.
            4. 답변은 반드시 한국어로 해 주세요.

            {context}

            Question: {question}
            Helpful Answer:
            """

    prompt_template = PromptTemplate.from_template(prompt_text)
    partial_template = prompt_template.partial(noResultMessage="답변 없음")

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | partial_template
        # | llm
        # | StrOutputParser()
    )

    message = "주식 시장이란 무엇인가?"

    # result = chain.invoke({"question": message})
    result = chain.invoke(message)

    print("___________result____________")
    print(result)
