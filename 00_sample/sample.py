import langchain
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

langchain.debug = True

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
삼성전자, 작년 영업이익 6조5000억원...전년 대비 84.9% ↓

9일 삼성전자는 지난해 연간 매출 258조1600억원, 영업이익 6조5400억원의 잠정 실적을 올렸다고 공시했다. 
2022년 매출 302조 2300억원, 영업이익 43조3800억원 대비 각각 14.58%, 84.92% 감소한 수치다. 
다만, 메모리 반도체 재고 감소 등으로 인한 반도체 업황 회복에 따라 실적이 빠르게 개선되고 있는 것으로 나타났다.

이날 회사는 지난해 4분기 매출 67조원, 영업이익 2조8000억원의 잠정 실적을 기록했다. 
전년 동기 대비 매출은 0.59% 감소하고 영업이익은 15.23% 증가했다. 
전년 동기 대비 실적이 고꾸라졌던 앞선 분기들과는 달리 4분기는 작년 수준으로 선방했다는 분석이 나온다.

실제 2023년 삼성전자의 분기별 실적은 빠르게 개선되는 모양새다. 
글로벌 경기 불황으로 삼성전자의 주력인 메모리 반도체 재고 증가와 IT 기기 수요 부진으로 영업이익이 2022년 4분기 4조3100억원에서 작년 1분기 6400억원으로 고꾸라진 뒤 
2분기 6700억원, 3분기 2조 4300억원, 4분기 2조8000억원으로 3개 분기 연속 실적이 개선되고 있다.

이같은 실적 개선은 반도체 업황 회복에 따른 것으로 분석된다. 
메모리 감산 효과가 나타나고 과잉 재고가 소진되면서 삼성전자의 주력인 반도체 실적이 개선되고 있는 것이다. 
실제로 삼성전자에서 반도체 사업을 하는 디바이스솔루션(DS) 부문은 작년 상반기 9조원이 넘는 적자를 기록하는 등 부진에 허덕였지만 4분기부터 본격적인 실적 개선이 이뤄진 것으로 추정된다. 
일부 증권가에서는 4분기 반도체 부문이 흑자 전환했을 것으로 예상하기도 한다.

DS(Device Solutions)부문 매출 28.56조원, 영업이익 6.45조원
메모리는 생성형 AI 서버용 제품의 수요 강세에 힘입어 시장 회복세가 지속되는 동시에, 기업용 자체 서버 시장의 수요도 증가하며 지난 분기에 이어 DDR5(Double Data Rate 5)와 고용량 SSD(Solid State Drive) 제품의 수요가 지속 확대되었다.
삼성전자는 ▲DDR5 ▲서버SSD ▲HBM(High Bandwidth Memory) 등 서버 응용 중심의 제품 판매 확대와 생성형 AI 서버용 고부가가치 제품 수요에 적극 대응해 실적이 전분기 대비 대폭 호전됐다.
또 업계 최초로 개발한 1b나노 32Gb DDR5 기반의 128GB 제품 양산 판매를 개시해 DDR5 시장 리더십을 강화했다.

시스템LSI는 주요 고객사 신제품용 SoC(System on Chip)·이미지센서·DDI(Display Driver IC) 제품 공급 증가로 실적이 개선돼 상반기 기준 역대 최대 매출을 달성했다.
파운드리는 시황 회복이 지연되는 상황에서도 5나노 이하 선단 공정 수주 확대로 전년 대비 AI와 고성능 컴퓨팅(HPC, High Performance Computing) 분야 고객수가 약 2배로 증가했다.
또 GAA(Gate All Around) 2나노 공정 프로세스 설계 키트 개발·배포를 통해 고객사들이 본격적으로 제품 설계를 진행 중이며, 2025년 2나노 양산을 위한 준비도 계획대로 추진하고 있다.

MX(Mobile eXperience)는 2분기 스마트폰 시장 비수기가 지속되면서 매출이 신모델이 출시된 1분기에 비해 감소했다. 판매호조가 지속되고 있는 S24 시리즈는 2분기와 상반기 출하량·매출 모두 전년 대비 두 자릿수 성장을 달성했다.
2분기에는 주요 원자재 가격 상승으로 인한 수익성 악화 요인이 있었으나 상반기 기준 두 자릿수 수익률을 유지했다.
VD(Visual Display)는 글로벌 대형 스포츠 이벤트 특수에 힘입어 선진 시장 성장을 중심으로 전년 대비 매출이 상승했다.
차별화된 2024년형 신모델 론칭을 기반으로 Neo QLED와 OLED, 라이프스타일 등 전략제품군 중심 판매에 주력해 프리미엄 시장 리더십을 강화했다.
생활가전은 성수기에 접어든 에어컨 제품 매출 확대와 비스포크 AI 신제품 판매 호조로 실적 회복세가 지속되고 있다.

하만은 포터블과 TWS(True Wireless Stereo) 중심의 소비자 오디오 제품 판매 확대로 실적이 개선됐다.
"""
chunk = """
이날 회사는 지난해 4분기 매출 67조원, 영업이익 2조8000억원의 잠정 실적을 기록했다. 
전년 동기 대비 매출은 0.59% 감소하고 영업이익은 15.23% 증가했다. 
전년 동기 대비 실적이 고꾸라졌던 앞선 분기들과는 달리 4분기는 작년 수준으로 선방했다는 분석이 나온다.
"""

template = ChatPromptTemplate.from_messages(
    [
        ("human", DOCUMENT_CONTEXT_PROMPT),
        ("human", CHUNK_CONTEXT_PROMPT),
    ]
)

# model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
# model = ChatOpenAI(temperature=0, model="gpt-4o")

chain = template | model
result = chain.invoke({"doc_content": doc, "chunk_content": chunk})
print(result)
