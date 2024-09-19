from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings()
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

query = "인터파크트리플의 쿠폰은 얼마까지 할인돼?"
docs = vector_store.similarity_search(query)
print(docs)

# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# retriever = vector_store.as_retriever()
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
# # prompt = ChatPromptTemplate.from_template(template)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )
#
# question_answer_chain = create_stuff_documents_chain(model, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#
# result = rag_chain.invoke(
#     {
#         "input": "인터파크트리플의 쿠폰은 얼마까지 할인돼?",
#         "question": "인터파크트리플의 쿠폰은 얼마까지 할인돼?",
#     }
# )
# print(result)
# {'input': '인터파크트리플의 쿠폰은 얼마까지 할인돼?', 'question': '인터파크트리플의 쿠폰은 얼마까지 할인돼?', 'context': [], 'answer': '인터파크트리플의 쿠폰은 최대 5,000원까지 할인됩니다.'}
