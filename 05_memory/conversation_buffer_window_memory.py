from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import OpenAI

load_dotenv()

conversation_with_summary = ConversationChain(
    llm=OpenAI(temperature=0),
    # k=2, to only keep the last 2 interactions in memory
    memory=ConversationBufferWindowMemory(k=1),
    verbose=True,
)
conversation_with_summary.predict(input="내 이름은 김고은이야.")
conversation_with_summary.predict(input="내 나이는 26살이야.")
conversation_with_summary.predict(input="나는 서울에 살아.")
result = conversation_with_summary.predict(input="내 이름은 뭐야?")
print(result)
# conversation_with_summary.predict(input="What's their issues?")


# memory = ConversationBufferWindowMemory(k=1, return_messages=True)
# memory.save_context({"input": "안녕"}, {"output": "안녕하세요"})
# memory.save_context({"input": "잘지내나요?"}, {"output": "그럭저럭"})
#
# variables = memory.load_memory_variables({})
# print(variables)

# memory = ConversationBufferWindowMemory(k=2, return_messages=True)
#
# memory.save_context(
#     inputs={
#         "human": "안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
#     },
#     outputs={
#         "ai": "안녕하세요! 계좌 개설을 원하신다니 기쁩니다. 먼저, 본인 인증을 위해 신분증을 준비해 주시겠어요?"
#     },
# )
# memory.save_context(
#     inputs={"human": "네, 신분증을 준비했습니다. 이제 무엇을 해야 하나요?"},
#     outputs={
#         "ai": "감사합니다. 신분증 앞뒤를 명확하게 촬영하여 업로드해 주세요. 이후 본인 인증 절차를 진행하겠습니다."
#     },
# )
# memory.save_context(
#     inputs={"human": "사진을 업로드했습니다. 본인 인증은 어떻게 진행되나요?"},
#     outputs={
#         "ai": "업로드해 주신 사진을 확인했습니다. 이제 휴대폰을 통한 본인 인증을 진행해 주세요. 문자로 발송된 인증번호를 입력해 주시면 됩니다."
#     },
# )
# memory.save_context(
#     inputs={"human": "인증번호를 입력했습니다. 계좌 개설은 이제 어떻게 하나요?"},
#     outputs={
#         "ai": "본인 인증이 완료되었습니다. 이제 원하시는 계좌 종류를 선택하고 필요한 정보를 입력해 주세요. 예금 종류, 통화 종류 등을 선택할 수 있습니다."
#     },
# )
# memory.save_context(
#     inputs={"human": "정보를 모두 입력했습니다. 다음 단계는 무엇인가요?"},
#     outputs={
#         "ai": "입력해 주신 정보를 확인했습니다. 계좌 개설 절차가 거의 끝났습니다. 마지막으로 이용 약관에 동의해 주시고, 계좌 개설을 최종 확인해 주세요."
#     },
# )
# memory.save_context(
#     inputs={"human": "모든 절차를 완료했습니다. 계좌가 개설된 건가요?"},
#     outputs={
#         "ai": "네, 계좌 개설이 완료되었습니다. 고객님의 계좌 번호와 관련 정보는 등록하신 이메일로 발송되었습니다. 추가적인 도움이 필요하시면 언제든지 문의해 주세요. 감사합니다!"
#     },
# )
#
# result = memory.load_memory_variables({})["history"]
# print(result)
