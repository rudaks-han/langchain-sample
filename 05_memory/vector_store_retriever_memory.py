from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

vectorstore = Chroma(collection_name="chroma_db", embedding_function=OpenAIEmbeddings())

from langchain.memory import VectorStoreRetrieverMemory

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
memory = VectorStoreRetrieverMemory(retriever=retriever)

memory.save_context(
    inputs={"human": "이번 프로젝트의 진행 상황을 업데이트해 주세요"},
    outputs={
        "ai": "네, 현재 개발 팀이 70% 정도 완료한 상태입니다. QA 팀은 다음 주부터 테스트를 시작할 예정입니다."
    },
)
memory.save_context(
    inputs={"human": "마케팅 전략은 어떻게 진행되고 있나요?"},
    outputs={
        "ai": "우리는 소셜 미디어 캠페인을 준비 중입니다. 첫 번째 단계는 다음 주 월요일에 시작될 예정입니다."
    },
)
memory.save_context(
    inputs={
        "human": "좋습니다. 다음 미팅은 두 주 후에 진행하죠. 그때까지 모든 팀이 진행 상황을 공유해 주세요."
    },
    outputs={"ai": "알겠습니다."},
)

# 메모리에 질문을 통해 가장 연관성 높은 1개 대화를 추출합니다.
print("____ 1 ____")
result = memory.load_memory_variables({"prompt": "프로젝트 진행상태는?"})["history"]
print(result)

print("____ 2 ____")
result = memory.load_memory_variables({"human": "마케팅 전략?"})["history"]
print(result)
