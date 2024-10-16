from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
Answer in KOREAN.
"""

doc = """
9일 삼성전자는 지난해 연간 매출 258조1600억원, 영업이익 6조5400억원의 잠정 실적을 올렸다고 공시했다. 
2022년 매출 302조 2300억원, 영업이익 43조3800억원 대비 각각 14.58%, 84.92% 감소한 수치다. 
다만, 메모리 반도체 재고 감소 등으로 인한 반도체 업황 회복에 따라 실적이 빠르게 개선되고 있는 것으로 나타났다.

이날 회사는 지난해 4분기 매출 67조원, 영업이익 2조8000억원의 잠정 실적을 기록했다. 
전년 동기 대비 매출은 0.59% 감소하고 영업이익은 15.23% 증가했다. 
전년 동기 대비 실적이 고꾸라졌던 앞선 분기들과는 달리 4분기는 작년 수준으로 선방했다는 분석이 나온다.
"""
chunk = """
이날 회사는 지난해 4분기 매출 67조원, 영업이익 2조8000억원의 잠정 실적을 기록했다. 
전년 동기 대비 매출은 0.59% 감소하고 영업이익은 15.23% 증가했다. 
전년 동기 대비 실적이 고꾸라졌던 앞선 분기들과는 달리 4분기는 작년 수준으로 선방했다는 분석이 나온다.
한국의 수도는 서울이다.
"""

template = ChatPromptTemplate.from_messages(
    [
        ("human", DOCUMENT_CONTEXT_PROMPT),
        ("human", CHUNK_CONTEXT_PROMPT),
    ]
)

model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

chain = template | model
result = chain.invoke({"doc_content": doc, "chunk_content": chunk})
print(result)
