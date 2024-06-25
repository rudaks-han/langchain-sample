from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()
# OpenAI의 "text-embedding-3-large" 모델을 사용하여 임베딩을 생성합니다.
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# embeddings = OpenAIEmbeddings()
#
# text = "LangChain 임베딩 테스트"

# query_result = embeddings.embed_query(text)
# query_result = embeddings.embed_documents(text)

# print(query_result[0][:3])

# sentence1 = "안녕하세요? 반갑습니다."
# sentence2 = "안녕하세요? 반갑습니다!"
# sentence3 = "안녕하세요? 만나서 반가워요."
# sentence4 = "Hi, nice to meet you."
# sentence5 = "I like to eat apples."
#
# from sklearn.metrics.pairwise import cosine_similarity

# embeddings = OpenAIEmbeddings()
# sentences = [sentence1, sentence2, sentence3, sentence4, sentence5]
# embedded_sentences = embeddings.embed_documents(sentences)


# def similarity(a, b):
#     return cosine_similarity([a], [b])[0][0]


# texts = [text1, text2]
# embedded_texts = embeddings.embed_documents(texts)
#
# print(embedded_texts)
# result = similarity(text1, text2)
# print(result)

# for i, sentence in enumerate(embedded_sentences):
#     for j, other_sentence in enumerate(embedded_sentences):
#         if i < j:
#             print(
#                 f"[유사도 {similarity(sentence, other_sentence):.4f}] {sentences[i]} \t <=====> \t {sentences[j]}"
#             )

from langchain.evaluation import load_evaluator

# evaluator = load_evaluator("embedding_distance")
embeddings = OpenAIEmbeddings()
evaluator = load_evaluator("pairwise_embedding_distance")


# text1 = "안녕하세요~"
# text2 = "안녕하세요!"
text1 = "사과가 너무 맛있어"
text2 = "오늘 비가 온다"
result = evaluator.evaluate_string_pairs(prediction=text1, prediction_b=text2)
print(result)
