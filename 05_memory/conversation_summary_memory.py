# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from langchain.memory import ConversationSummaryMemory, ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

# API KEY 정보로드
load_dotenv()

llm = ChatOpenAI(temperature=0, verbose=True)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100,
    return_messages=True,
)

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

result = memory.load_memory_variables({})["history"]
print(result)


# A: 이번 프로젝트의 진행 상황을 업데이트해 주세요.
# B: 네, 현재 개발 팀이 70% 정도 완료한 상태입니다. QA 팀은 다음 주부터 테스트를 시작할 예정입니다.
# A: 마케팅 전략은 어떻게 진행되고 있나요?
# C: 우리는 소셜 미디어 캠페인을 준비 중입니다. 첫 번째 단계는 다음 주 월요일에 시작될 예정입니다.
# A: 좋습니다. 다음 미팅은 두 주 후에 진행하죠. 그때까지 모든 팀이 진행 상황을 공유해 주세요.


def conversation_summary_memory():
    memory = ConversationSummaryMemory(llm=llm, return_messages=True)

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

    result = memory.load_memory_variables({})["history"]
    print(result)


def conversation_summary_buffer_memory():
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=200,  # 요약의 기준이 되는 토큰 길이를 설정합니다.
        return_messages=True,
    )

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

    result = memory.load_memory_variables({})["history"]
    print(result)


# if __name__ == "__main__":
# conversation_summary_memory()
# conversation_summary_buffer_memory()
