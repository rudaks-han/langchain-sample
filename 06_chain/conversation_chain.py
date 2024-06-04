from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()

llm = OpenAI(temperature=0)

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
)

result = conversation.predict(input="Hi there!")
print(result)

result = conversation.predict(input="What's the weather?")
print(result)


def conversation():
    # ConversationChain 인스턴스를 생성합니다.
    # llm: 모델을 지정합니다.
    # verbose: 상세한 로깅을 비활성화합니다.
    # memory: 대화 내용을 저장하는 메모리 버퍼를 지정합니다.
    conversation = ConversationChain(
        llm=llm,
        verbose=False,
        memory=ConversationBufferMemory(memory_key="history"),
    )

    conversation.invoke({"input": "양자역학에 대해 설명해줘."})

    conversation.memory.load_memory_variables({})["history"]

    conversation.memory.save_context(inputs={"human": "hi"}, outputs={"ai": "안녕"})
    conversation.memory.load_memory_variables({})["history"]

    print(
        conversation.invoke({"input": "불렛포인트 형식으로 작성해줘. emoji 추가해줘."})
    )


from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI


def conversation_streaming():
    class StreamingHandler(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            print(f"{token}", end="", flush=True)

    # 스트리밍을 활성화하기 위해, ChatModel 생성자에 `streaming=True`를 전달합니다.
    # 추가적으로, 사용자 정의 핸들러 리스트를 전달합니다.
    stream_llm = ChatOpenAI(
        model="gpt-4-turbo-preview", streaming=True, callbacks=[StreamingHandler()]
    )

    conversation = ConversationChain(
        llm=stream_llm,
        verbose=False,
        memory=ConversationBufferMemory(),
    )

    output = conversation.predict(input="양자역학에 대해 설명해줘")

    output = conversation.predict(
        input="이전의 내용을 불렛포인트로 요약해줘. emoji 추가해줘."
    )


def prompt_tuning():
    from langchain.prompts import PromptTemplate

    template = """
    당신은 10년차 엑셀 전문가 입니다. 아래 대화내용을 보고 질문에 대한 적절한 답변을 해주세요
    
    #대화내용
    {chat_history}
    ----
    사용자: {question}
    엑셀전문가:"""

    prompt = PromptTemplate.from_template(template)

    prompt.partial(chat_history="엑셀에서 데이터를 필터링하는 방법에 대해 알려주세요.")

    class StreamingHandler(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            print(f"{token}", end="", flush=True)

    stream_llm = ChatOpenAI(
        model="gpt-4-turbo-preview", streaming=True, callbacks=[StreamingHandler()]
    )
    conversation = ConversationChain(
        llm=stream_llm,
        prompt=prompt,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        input_key="question",
    )

    answer = conversation.predict(
        question="엑셀에서 VLOOKUP 함수는 무엇인가요? 간단하게 설명해주세요"
    )

    answer = conversation.predict(question="예제를 보여주세요")
    print(answer)


# if __name__ == "__main__":
# conversation()
# conversation_streaming()
# prompt_tuning()
