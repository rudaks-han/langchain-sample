import asyncio
from typing import Any

from dotenv import load_dotenv
from langchain.agents import (
    AgentExecutor,
    create_openai_functions_agent,
)
from langchain.callbacks import (
    AsyncIteratorCallbackHandler,
)
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from handlers.chat_model_start_handler import ChatModelStartHandler
from tools.sql import run_query_tool, list_tables, describe_tables_tool

load_dotenv()


class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False

    def __init__(self) -> None:
        super().__init__()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token
        # if we passed the final answer, we put tokens in queue
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ['"', "}"]:
                    self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""


handler = ChatModelStartHandler()
# chat = ChatOpenAI(callbacks=[handler], streaming=True)
# chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
chat = ChatOpenAI(streaming=True, callbacks=[AsyncCallbackHandler()])

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
agent_executor.callbacks = [AsyncCallbackHandler()]


async def print_result(_agent_executor):
    chunks = []

    async for chunk in _agent_executor.astream(
        {"input": "How many users are in the database?"}
    ):
        print("____ chunk ____")
        print(chunk)
        # print("chunk:", chunk)
        # chunks.append(chunk)
        # yield chunk
        # if "output" in chunk:
        #     print(chunk)
        # for char in chunk["output"]:
        #     print(char, end="\n", flush=True)

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
