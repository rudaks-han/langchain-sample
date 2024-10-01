from dotenv import load_dotenv
from langchain_upstage import UpstageGroundednessCheck

load_dotenv()

groundedness_check = UpstageGroundednessCheck()

request_input = {
    "context": "삼성전자는 연결 기준으로 매출 74.07조원, 영업이익 10.44조원의 2024년 2분기 실적을 발표했다. 전사 매출은 전분기 대비 3% 증가한 74.07조원을 기록했다. DS부문은 메모리 업황 회복으로 전분기 대비 23% 증가하고, SDC는 OLED 판매 호조로 증가했다.",
    "answer": "삼성전자의 2024년 전체 매출은 약 74조원이다.",
}

response = groundedness_check.invoke(request_input)
print(response)  # notGrounded
