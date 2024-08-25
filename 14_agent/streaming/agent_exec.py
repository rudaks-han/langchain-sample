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
from langchain_core.callbacks import StreamingStdOutCallbackHandler, BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from sql import run_query_tool, list_tables, describe_tables_tool

load_dotenv()


class MyCallbackHandler(BaseCallbackHandler):

    def on_llm_new_token(self, token, **kwargs) -> None:
        print(f"{token}")


class CallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.content: str = ""
        self.final_answer: bool = False

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.content += token
        if "Final Answer" in self.content:
            # now we're in the final answer section, but don't print yet
            self.final_answer = True
            self.content = ""
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ["}"]:
                    print(token, end="")
                    # sys.stdout.write(token)  # equal to `print(token, end="")`
                    # sys.stdout.flush()


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


def get_agent_executor():
    print("get_agent_executor start")
    llm = ChatOpenAI(
        streaming=True,
        callbacks=[
            # FinalStreamingStdOutCallbackHandler(
            #     answer_prefix_tokens=["Final", "Answer"]
            # )
            # FinalStreamingStdOutCallbackHandler()
            # FinalStreamingStdOutCallbackHandler(answer_prefix_tokens=["The", "answer", ":"])
            # MyCallbackHandler()
            # CallbackHandler()
            # AsyncCallbackHandler()
        ],
        # callbacks=[StreamingStdOutCallbackHandler()],
    )

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
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    # agent.agent.llm_chain.llm.callbacks = [AsyncCallbackHandler()]

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=3,
        return_intermediate_steps=False,
        early_stopping_method="generate",
        verbose=True,
        stream_prefix=True,
    )
    # agent_executor.callbacks = [CallbackHandler()]

    print("get_agent_executor return")
    return agent_executor


# agent_executor.invoke({"input": "How many users are in the database?"})
# response = agent_executor.invoke(
#     {"input": "How many users have provided a shipping addresses?"}
# )
# print(response)


# async def print_result():
#     async for chunk in agent_executor.astream(
#         {"input": "How many users have provided a shipping addresses?"}
#     ):
#         print(chunk, flush=True)
#
#
# asyncio.run(print_result())

# agent = initialize_agent(
#     agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
#     tools=tools,
#     llm=llm,
#     verbose=True,
#     max_iterations=3,
#     early_stopping_method="generate",
#     return_intermediate_steps=False,
# )
# agent_response = agent({"input": "How many users have provided a shipping addresses?"})
# print(agent_response)
