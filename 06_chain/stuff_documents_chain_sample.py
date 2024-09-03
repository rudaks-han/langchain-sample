# !pip install -U langchain langchainhub langchain_openai langchain_community -q
import langchain
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# API 키 정보 로드
load_dotenv()
langchain.debug = True
prompt = ChatPromptTemplate.from_messages(
    ["Please summarize the following documents: {context2}"]
)

# retriever를 통해 docs를 리턴받는다. (여기서 retriever는 생략)
docs = [
    Document(page_content="신용카드 발급 방법은"),
    Document(page_content="개인정보에 동의하면 됩니다."),
]

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.0,
)
chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context2")
answer = chain.invoke({"context2": docs})
print("answer:", answer)
