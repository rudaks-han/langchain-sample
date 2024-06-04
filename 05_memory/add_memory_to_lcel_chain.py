# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

# API KEY 정보로드
load_dotenv()

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a chatbot having a conversation with a human."
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # Where the human input will injected
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI()

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

chat_llm_chain.predict(human_input="Hi there my friend")
chat_llm_chain.predict(human_input="Not too bad - how are you?")


# model = ChatOpenAI()
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful chatbot"),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}"),
#     ]
# )
#
# # 대화 버퍼 메모리를 생성하고, 메시지 반환 기능을 활성화합니다.
# memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
#
# memory.load_memory_variables({})  # 메모리 변수를 빈 딕셔너리로 초기화합니다.
#
# runnable = RunnablePassthrough.assign(
#     chat_history=RunnableLambda(memory.load_memory_variables)
#     | itemgetter("chat_history")  # memory_key 와 동일하게 입력합니다.
# )
#
# runnable.invoke({"input": "hi!"})
#
# chain = runnable | prompt | model
#
# # chain 객체의 invoke 메서드를 사용하여 입력에 대한 응답을 생성합니다.
# response = chain.invoke({"input": "만나서 반갑습니다. 제 이름은 테디입니다."})
# print(response)  # 생성된 응답을 출력합니다.
#
# # 입력된 데이터와 응답 내용을 메모리에 저장합니다.
# memory.save_context(
#     {"inputs": "만나서 반갑습니다. 제 이름은 테디입니다."}, {"output": response.content}
# )
#
# # 저장된 대화기록을 출력합니다.
# memory.load_memory_variables({})
#
# # 이름을 기억하고 있는지 추가 질의합니다.
# response = chain.invoke({"input": "제 이름이 무엇이었는지 기억하세요?"})
# # 답변을 출력합니다.
# print(response.content)
