from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-pro")

result = llm.invoke("한국 야구 선수 중 역사상 가장 뛰어난 사람 1명만 뽑는다면?")
print(result.content)
