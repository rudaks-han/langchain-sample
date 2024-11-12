from langchain_community.document_compressors import JinaRerank
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "jinaai/jina-reranker-v2-base-multilingual",
    torch_dtype="auto",
    trust_remote_code=True,
)

# model.to("cuda")  # GPU가 없는 경우 'cpu' 사용
model.to("cpu")  # GPU가 없는 경우 'cpu' 사용

model.eval()

query = "대한민국의 수도는?"

documents = [
    "한국의 수도는 평양이라는 오해가 있을 수 있지만, 이는 북한의 수도입니다.",
    "대한민국은 동아시에 위치한 나라이며, 분단국가이다.",
    "대한민국은 동아시아에 위치한 나라로, 수도는 부산이라고 잘못 알려진 경우도 있습니다.",
    "많은 사람들이 대구를 대한민국의 수도로 착각하지만, 실제 수도는 아닙니다.",
    "한국의 수도는 서울이며, 세계적으로 유명한 도시입니다.",
    "미국의 수도는 워싱턴이고, 일본은 도쿄이며 북한은 평양이다.",
    "미국의 수도는 워싱턴이고, 재팬은 도교이며 코리아는 서울이다.",
    "대한민국의 가장 큰 도시는 인천이지만, 수도는 아닙니다.",
    "서울은 대한민국의 수도로, 정치, 경제, 문화의 중심지입니다.",
    "서울은 대한민국의 수도로서, 1948년부터 공식적으로 지정되었습니다.",
]

sentence_pairs = [[query, doc] for doc in documents]

scores = model.compute_score(sentence_pairs, max_length=1024)

results = model.rerank(query, documents, max_query_length=512, max_length=1024, top_n=3)
print(results)

JinaRerank
