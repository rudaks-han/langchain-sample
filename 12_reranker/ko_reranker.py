import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


model_path = "Dongjin-kr/ko-reranker"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

pairs = [
    ["나는 너를 싫어해", "나는 너를 사랑해"],
    ["나는 너를 좋아해", "너에 대한 나의 감정은 사랑 일 수도 있어"],
]

pairs = [
    [
        "대한민국의 수도는?",
        "한국의 수도는 평양이라는 오해가 있을 수 있지만, 이는 북한의 수도입니다.",
    ],
    ["대한민국의 수도는?", "대한민국은 동아시에 위치한 나라이며, 분단국가이다."],
    [
        "대한민국의 수도는?",
        "대한민국은 동아시아에 위치한 나라로, 수도는 부산이라고 잘못 알려진 경우도 있습니다.",
    ],
    [
        "대한민국의 수도는?",
        "많은 사람들이 대구를 대한민국의 수도로 착각하지만, 실제 수도는 아닙니다.",
    ],
    ["대한민국의 수도는?", "한국의 수도는 서울이며, 세계적으로 유명한 도시입니다."],
    [
        "대한민국의 수도는?",
        "미국의 수도는 워싱턴이고, 일본은 도쿄이며 북한은 평양이다.",
    ],
    [
        "대한민국의 수도는?",
        "미국의 수도는 워싱턴이고, 재팬은 도교이며 코리아는 서울이다.",
    ],
    ["대한민국의 수도는?", "대한민국의 가장 큰 도시는 인천이지만, 수도는 아닙니다."],
    [
        "대한민국의 수도는?",
        "서울은 대한민국의 수도로, 정치, 경제, 문화의 중심지입니다.",
    ],
    [
        "대한민국의 수도는?",
        "서울은 대한민국의 수도로서, 1948년부터 공식적으로 지정되었습니다.",
    ],
]

with torch.no_grad():
    inputs = tokenizer(
        pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    scores = (
        model(**inputs, return_dict=True)
        .logits.view(
            -1,
        )
        .float()
    )
    scores = exp_normalize(scores.numpy())

print(np.round(scores * 100, 2))

# with torch.no_grad():
#     inputs = tokenizer(
#         pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
#     )
#     scores = (
#         model(**inputs, return_dict=True)
#         .logits.view(
#             -1,
#         )
#         .float()
#     )
#     scores = exp_normalize(scores.numpy())
#     print(f"first: {scores[0]}, second: {scores[1]}")
