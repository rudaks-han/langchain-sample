from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


llm = OllamaLLM(model="llama3.1-instruct-8b")


template = "{country}의 수도는?"

prompt = PromptTemplate.from_template(template=template)
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"country", "한국"})
print(result)
