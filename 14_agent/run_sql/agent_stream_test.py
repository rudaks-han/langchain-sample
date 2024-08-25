import os
import sys

from langchain.agents import (
    load_tools,
    create_openai_functions_agent,
    AgentExecutor,
)
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI


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
                    sys.stdout.write(token)  # equal to `print(token, end="")`
                    sys.stdout.flush()


llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
    model_name="gpt-3.5-turbo",
    streaming=True,  # ! important
    callbacks=[StreamingStdOutCallbackHandler()],  # ! important
)


# create messages to be passed to chat LLM

# initialize conversational memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True, output_key="output"
)

# create a single tool to see how it impacts streaming
tools = load_tools(["llm-math"], llm=llm)

# initialize the agent
# agent = initialize_agent(
#     agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
#     tools=tools,
#     llm=llm,
#     memory=memory,
#     # verbose=True,
#     # max_iterations=3,
#     # early_stopping_method="generate",
#     # return_intermediate_steps=False,
# )
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
# agent.llm.callbacks = [CallbackHandler()]

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=3,
    return_intermediate_steps=False,
    early_stopping_method="generate",
    verbose=False,
)
agent_executor.callbacks = [CallbackHandler()]

response = agent_executor.invoke({"input": "tell me a short story"})
print(response)
