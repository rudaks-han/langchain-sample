from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAI

load_dotenv()

llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)

with get_openai_callback() as callback:
    question = "3x2는 몇이야?"
    result = conversation.predict(input=question)
    print(result)

with get_openai_callback() as callback:
    question = "거기에 2를 곱하면?"
    result = conversation.predict(input=question)
    print(result)


# memory = ConversationBufferMemory(return_messages=True)
# memory.save_context({"input": "안녕"}, {"output": "안녕하세요!"})
#
# variables = memory.load_memory_variables({})
# print(variables)

# memory = ConversationBufferMemory()
# memory.save_context(
#     inputs={
#         "human": "안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
#     },
#     outputs={
#         "ai": "안녕하세요! 계좌 개설을 원하신다니 기쁩니다. 먼저, 본인 인증을 위해 신분증을 준비해 주시겠어요?"
#     },
# )
#
# # 'history' 키에 저장된 대화 기록을 확인합니다.
# memory.load_memory_variables({})
#
# memory.save_context(
#     inputs={"human": "네, 신분증을 준비했습니다. 이제 무엇을 해야 하나요?"},
#     outputs={
#         "ai": "감사합니다. 신분증 앞뒤를 명확하게 촬영하여 업로드해 주세요. 이후 본인 인증 절차를 진행하겠습니다."
#     },
# )
#
# # 2개의 대화를 저장합니다.
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
#
# # history에 저장된 대화 기록을 확인합니다.
# # print(memory.load_memory_variables({})["history"])
#
# # 추가로 2개의 대화를 저장합니다.
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
# # history에 저장된 대화 기록을 확인합니다.
# # print(memory.load_memory_variables({})["history"])
#
# memory = ConversationBufferMemory(return_messages=True)
#
# memory.save_context(
#     inputs={
#         "human": "안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
#     },
#     outputs={
#         "ai": "안녕하세요! 계좌 개설을 원하신다니 기쁩니다. 먼저, 본인 인증을 위해 신분증을 준비해 주시겠어요?"
#     },
# )
#
# memory.save_context(
#     inputs={"human": "네, 신분증을 준비했습니다. 이제 무엇을 해야 하나요?"},
#     outputs={
#         "ai": "감사합니다. 신분증 앞뒤를 명확하게 촬영하여 업로드해 주세요. 이후 본인 인증 절차를 진행하겠습니다."
#     },
# )
#
# memory.save_context(
#     inputs={"human": "사진을 업로드했습니다. 본인 인증은 어떻게 진행되나요?"},
#     outputs={
#         "ai": "업로드해 주신 사진을 확인했습니다. 이제 휴대폰을 통한 본인 인증을 진행해 주세요. 문자로 발송된 인증번호를 입력해 주시면 됩니다."
#     },
# )
#
# memory_history = memory.load_memory_variables({})["history"]
# print(memory_history)
#
# # API KEY를 환경변수로 관리하기 위한 설정 파일
# from dotenv import load_dotenv
#
# # API KEY 정보로드
# load_dotenv()
#
# from langchain_openai import ChatOpenAI
# from langchain.chains import ConversationChain
#
# # LLM 모델을 생성합니다.
# llm = ChatOpenAI(temperature=0)
#
# # ConversationChain을 생성합니다.
# conversation = ConversationChain(
#     # ConversationBufferMemory를 사용합니다.
#     llm=llm,
#     memory=ConversationBufferMemory(),
# )
#
# # 대화를 시작합니다.
# response = conversation.predict(
#     input="안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
# )
# print(response)
#
# # 이전 대화내용을 불렛포인트로 정리해 달라는 요청을 보냅니다.
# response = conversation.predict(
#     input="이전 답변을 bullet point 형식으로 정리하여 알려주세요."
# )
# print("___ 이전정보 기억하는지 확인")
# print(response)
