from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

docs = [
    Document(
        page_content='사용자가 로그인 시도를 할 때 "500 Internal Server Error'
        '" 메시지가 나타납니다. 특정 브라우저에서 더 자주 발생합니다.',
        metadata={
            "project_name": "국민은행",
            "issue_type": "bug",
            "priority": "high",
            "assignee": "홍길동",
            "job_days": 5,
        },
    ),
    Document(
        page_content="사용자가 실시간 데이터 분석을 할 수 있는 새로운 대시보드 위젯을 추가합니다. 필요한 데이터는 사용자의 현재 세션에서 가져옵니다.",
        metadata={
            "project_name": "신한은행",
            "issue_type": "story",
            "priority": "middle",
            "assignee": "김철수",
            "job_days": 3,
        },
    ),
    Document(
        page_content="현재 프로필 사진 업데이트가 너무 느리고 자주 실패합니다. 업로드 속도 향상 및 실패율 감소가 필요합니다.",
        metadata={
            "project_name": "국민은행",
            "issue_type": "improvement",
            "priority": "low",
            "assignee": "이영희",
            "job_days": 1,
        },
    ),
    Document(
        page_content="매주 월요일 오전 9시에 자동으로 주간 보고서를 생성하고 관련 부서에 이메일로 전송되도록 설정합니다.",
        metadata={
            "project_name": "하나은행",
            "issue_type": "task",
            "priority": "middle",
            "assignee": "박민수",
            "job_days": 4,
        },
    ),
    Document(
        page_content="새로운 결제 게이트웨이 API를 기존 시스템에 통합하여 결제 처리 속도와 신뢰성을 높입니다. 메인 태스크: 결제 시스템 업그레이드.",
        metadata={
            "project_name": "카카오뱅크",
            "issue_type": "task",
            "priority": "high",
            "assignee": "최지은",
            "job_days": 5,
        },
    ),
]
vectorstore = Chroma.from_documents(
    persist_directory="chroma_self_query_store",
    documents=docs,
    embedding=OpenAIEmbeddings(),
)
