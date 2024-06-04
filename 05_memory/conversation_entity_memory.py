import langchain
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_openai import OpenAI

langchain.debug = True
load_dotenv()

llm = OpenAI(temperature=0)

# memory = ConversationEntityMemory(llm=llm)
# _input = {
#     "input": "안녕하세요, 저는 한유주입니다. 제가 지난주에 갔던 파리 여행에 대해 이야기하고 싶어요. 에펠탑에서 찍은 멋진 사진들도 있어요."
# }
# memory.load_memory_variables(_input)
# memory.save_context(
#     _input,
#     {
#         "output": "안녕하세요, 한유주님! 파리 여행이라니 정말 멋지네요. 에펠탑에서 찍은 사진들 중 가장 마음에 드는 사진은 어떤 것인가요"
#     },
# )

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm),
)

conversation.predict(
    input="안녕하세요, 저는 한유주입니다. 제가 지난주에 갔던 파리 여행에 대해 이야기하고 싶어요. 에펠탑에서 찍은 멋진 사진들도 있어요"
)


result = conversation.invoke(input="유주에 대해 뭘 알고 있어?")

print(result)

# memory = ConversationEntityMemory(llm=llm, return_messages=True)
# _input = {"input": "은규와 지훈은 탁구를 치고 있어"}
# memory.load_memory_variables(_input)
# memory.save_context(
#     _input,
#     {"output": "재미있겠다. 누가 이기고 있어?"},
# )
#
# variables = memory.load_memory_variables({"input": "은규는 누구야?"})
# print(variables)
# variable = memory.load_memory_variables({"input": "탁구는 누가 치고 있어?"})
# print(variable)
