from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

question = "대출 이자"
sentences = [
    "돈 빌리는 이자",
    "은행 이자",
    "대출하는데 이자",
    "나는 학생이다",
    "예금 이자",
    "대출 이자",
]
# question = "빨간 사과"
# sentences = [
#     "붉은 사과",
#     "노란 사과",
#     "red apple",
# ]

from sklearn.metrics.pairwise import cosine_similarity

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

embedded_sentences = embeddings.embed_documents(sentences)
embedded_question = embeddings.embed_query(question)


def similarity(a, b):
    return cosine_similarity([a], [b])[0][0]


for i, embedded_sentence in enumerate(embedded_sentences):
    print(
        f"[유사도 {similarity(embedded_question, embedded_sentence):.4f}] {question} \t <=====> \t {sentences[i]}"
    )

# for i, sentence in enumerate(embedded_sentences):
#     for j, other_sentence in enumerate(embedded_sentences):
#         if i < j:
#             print(
#                 f"[유사도 {similarity(sentence, other_sentence):.4f}] {sentences[i]} \t <=====> \t {sentences[j]}"
#             )
