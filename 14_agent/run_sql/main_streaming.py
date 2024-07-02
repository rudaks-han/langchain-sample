import asyncio

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
# chat = ChatOpenAI(callbacks=[handler], streaming=True)
chat = ChatOpenAI(streaming=True)

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


async def print_result(_agent_executor):
    chunks = []

    async for chunk in _agent_executor.astream(
        {"input": "How many users are in the database?"}
    ):
        print("chunk:", chunk)
        # chunks.append(chunk)
        # yield chunk

    # async for chunk in _agent_executor.astream(
    #     {"input": "How many users are in the database?"}
    # ):
    # print(chunk, end="|", flush=True)
    # print("___")
    # print(chunk)
    # Agent Action

    # if "actions" in chunk:
    #     pass
    #     # for action in chunk["actions"]:
    #     #     print(f"Calling Tool: `{action.tool}` with input `{action.tool_input}`")
    # # Observation
    # elif "steps" in chunk:
    #     pass
    #     # for step in chunk["steps"]:
    #     #     print(f"Tool Result: `{step.observation}`")
    # # Final result
    # elif "output" in chunk:
    #     print(f'Final Output: {chunk["output"]}', flush=True)
    # else:
    #     raise ValueError()
    # print("---")


def main():
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(print_result(agent_executor))
    # loop.close()
    asyncio.run(print_result(agent_executor))


if __name__ == "__main__":
    main()
