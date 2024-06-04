from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# llm = ChatOpenAI()
llm = ChatOpenAI(
    temperature=0,  # 창의성 (0.0 ~ 2.0)
    max_tokens=1024,  # 최대 토큰수
    model_name="gpt-4o",  # 모델명
)

template = "{country} 야구 선수 중 역사상 가장 뛰어난 사람 1명만 뽑는다면?"

prompt = PromptTemplate.from_template(template=template)
chain = LLMChain(prompt=prompt, llm=llm)

result = chain.invoke({"country", "한국"})
print(result)
