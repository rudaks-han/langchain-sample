# pipenv install datasets

from langchain_community.document_loaders import HuggingFaceDatasetLoader

dataset_name = "imdb"  # 데이터셋 이름을 "imdb"로 설정합니다.
page_content_column = "text"  # 페이지 내용이 포함된 열의 이름을 "text"로 설정합니다.

# HuggingFaceDatasetLoader를 사용하여 데이터셋을 로드합니다.
# 데이터셋 이름과 페이지 내용 열 이름을 전달합니다.
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

data = loader.load()  # 로더를 사용하여 데이터를 불러옵니다.

print(data[:3])

from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders.hugging_face_dataset import (
    HuggingFaceDatasetLoader,
)

dataset_name = "tweet_eval"  # 데이터셋 이름을 "tweet_eval"로 설정합니다.
page_content_column = "text"  # 페이지 내용이 포함된 열의 이름을 "text"로 설정합니다.
name = "stance_climate"  # 데이터셋의 특정 부분을 식별하는 이름을 "stance_climate"로 설정합니다.

# HuggingFaceDatasetLoader를 사용하여 데이터셋을 로드합니다.
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column, name)

# 로더에서 벡터 저장소 인덱스를 생성합니다.
index = VectorstoreIndexCreator().from_loaders([loader])

query = "What are the most used hashtag?"  # 가장 많이 사용되는 해시태그는 무엇인가요?
result = index.query(query)  # 질의를 수행하여 결과를 얻습니다.
print(result)
