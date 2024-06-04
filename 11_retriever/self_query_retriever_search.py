import langchain
from dotenv import load_dotenv
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

langchain.debug = True

load_dotenv()

# metadata={"issue_type": "bug", "priority": "high", "assignee": "홍길동"},
metadata_field_info = [
    AttributeInfo(
        name="project_name",
        description="프로젝트의 이름을 나타낸다.",
        type="string",
    ),
    AttributeInfo(
        name="issue_type",
        description="issue의 유형을 나타낸다. One of ['bug', 'story', 'improvement', 'task']",
        type="string",
    ),
    AttributeInfo(
        name="priority",
        description="이슈의 우선순위",
        type="string",
    ),
    AttributeInfo(
        name="assignee",
        description="이슈의 할당자 이름",
        type="string",
    ),
    AttributeInfo(
        name="job_days",
        description="이슈 작업에 걸리는 공수",
        type="integer",
    ),
]

document_content_description = "이슈의 내용"

# LLM 정의
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

vectorstore = Chroma(
    persist_directory="chroma_self_query_store", embedding_function=OpenAIEmbeddings()
)

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)

# docs = retriever.invoke("공수가 4 이상인 이슈를 찾아주세요")
docs = retriever.invoke("국민은행의 버그를 찾아주세요")
for doc in docs:
    print(doc.page_content)
