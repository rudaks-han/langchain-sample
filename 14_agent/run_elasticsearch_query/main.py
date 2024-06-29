from dotenv import load_dotenv
from langchain.agents import (
    AgentExecutor,
    create_openai_functions_agent,
)
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI

from handlers.chat_model_start_handler import ChatModelStartHandler
from tools.search import run_query_tool

load_dotenv()

handler = ChatModelStartHandler()
chat = ChatOpenAI(callbacks=[handler])

# print(tables)
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage("You are an AI that has access to s Elasticsearch.\n"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [run_query_tool]
agent = create_openai_functions_agent(llm=chat, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
# agent_executor.invoke({"input": "How many users are in the database?"})
agent_executor.invoke({"input": "에러 로그 보여줘"})
