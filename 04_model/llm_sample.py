from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(temperature=0)

template = "{country}의 수도는?"
prompt = PromptTemplate(template=template)
chain = prompt | model | StrOutputParser()
response = chain.invoke({"country": "한국"})
print(response)
