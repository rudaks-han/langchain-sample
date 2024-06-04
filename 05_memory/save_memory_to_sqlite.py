# %pip install -qU langchain-community SQLAlchemy langchain-openai

# os.environ["LANGCHAIN_TRACING_V2"] = "true"  # LANGCHAIN_TRACING_V2 환경 변수를 "true"로 설정합니다.
# os.environ["LANGCHAIN_API_KEY"] = "LANGCHAIN_API_KEY를 설정합니다."

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

from langchain_community.chat_message_histories import SQLChatMessageHistory

# SQLChatMessageHistory 객체를 생성하고 세션 ID와 데이터베이스 연결 문자열을 전달합니다.
chat_message_history = SQLChatMessageHistory(
    session_id="sql_chat_history", connection_string="sqlite:///sqlite.db"
)

# 사용자 메시지를 추가합니다.
chat_message_history.add_user_message(
    "Hi! My name is Teddy. I am a AI programmer. Nice to meet you!"
)
# AI 메시지를 추가합니다.
chat_message_history.add_ai_message("Hi Teddy! Nice to meet you too!")

chat_message_history.messages

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        # 시스템 메시지를 설정하여 어시스턴트의 역할을 정의합니다.
        ("system", "You are a helpful assistant."),
        # 이전 대화 내용을 포함하기 위한 플레이스홀더를 추가합니다.
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),  # 사용자의 질문을 입력받는 메시지를 설정합니다.
    ]
)

chain = (
    prompt | ChatOpenAI()
)  # 프롬프트와 ChatOpenAI 모델을 연결하여 체인을 생성합니다.

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///sqlite.db"
    ),  # session_id를 기반으로 SQLChatMessageHistory 객체를 생성하는 람다 함수
    input_messages_key="question",  # 입력 메시지의 키를 "question"으로 설정
    history_messages_key="history",  # 대화 기록 메시지의 키를 "history"로 설정
)

# 세션 ID를 구성하는 곳입니다.
config = {"configurable": {"session_id": "sql_chat_history"}}

# 질문 "Whats my name"과 설정을 사용하여 대화 기록이 있는 체인을 호출합니다.
response = chain_with_history.invoke({"question": "Whats my name?"}, config=config)
print(response.content)  # 응답을 출력합니다.
