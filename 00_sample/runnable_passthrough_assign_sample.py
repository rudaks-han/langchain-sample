from langchain.schema.runnable import RunnablePassthrough

# 예시 데이터
input_data = {"name": "Alice", "age": 30}

# assign을 사용해 동적으로 필드 추가 및 수정
runnable = RunnablePassthrough().assign(greeting=lambda x: f"Hello, {x['name']}!")

# 결과 출력
result = runnable.invoke(input_data)
print(result)

import time
from langchain.schema.runnable import RunnablePassthrough


# 시간 기반으로 데이터를 추가하는 함수
def get_timestamp(_):
    return time.time()


# 기본 입력 데이터
input_data = {"name": "Bob", "age": 25}

# assign을 통해 필드 업데이트
runnable = RunnablePassthrough().assign(timestamp=get_timestamp)

# 결과 출력
result = runnable.invoke(input_data)
print(result)


from langchain.schema.runnable import RunnablePassthrough


# 사용자 데이터를 받아 동적으로 이메일 도메인 생성
def extract_email_domain(data):
    email = data["email"]
    return email.split("@")[-1]


# 사용자 정보
input_data = {"name": "Charlie", "email": "charlie@example.com", "age": 28}

# assign을 사용해 동적으로 도메인 필드를 추가
runnable = RunnablePassthrough().assign(domain=extract_email_domain)

# 결과 출력
result = runnable.invoke(input_data)
print(result)
