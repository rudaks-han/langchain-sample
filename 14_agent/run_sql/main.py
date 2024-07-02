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
from tools.sql import run_query_tool, list_tables, describe_tables_tool

load_dotenv()

handler = ChatModelStartHandler()
chat = ChatOpenAI(callbacks=[handler])

tables = list_tables()
# print(tables)
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            "You are an AI that has access to s SQLite database.\n"
            f"The database has table of : {tables}\n"
            "Do not make any assumptions about what tables exist "
            "or what columns exists. Instead, use the 'describe_tables' function "
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [run_query_tool, describe_tables_tool]
agent = create_openai_functions_agent(llm=chat, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
# agent_executor.invoke({"input": "How many users are in the database?"})
response = agent_executor.invoke(
    {"input": "How many users have provided a shipping addresses?"}
)
print(response)
