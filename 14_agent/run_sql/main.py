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

from tools.sql import run_query_tool, list_tables, describe_tables_tool

# langchain.debug = True

load_dotenv()

chat = ChatOpenAI()

tables = list_tables()
# print(tables)
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage("You are an AI that has access to s SQLite database.\n{tables}"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [run_query_tool, describe_tables_tool]
agent = create_openai_functions_agent(llm=chat, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)
# agent_executor.invoke({"input": "How many users are in the database?"})
agent_executor.invoke({"input": "How many users have provided a shipping addresses?"})
