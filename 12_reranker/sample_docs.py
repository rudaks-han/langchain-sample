import time
from functools import wraps

from langchain_core.documents import Document


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"[doc {i+1}]" + d.page_content for i, d in enumerate(docs)]
        )
    )


def pretty_print_texts(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"[doc {i+1}]" + d["text"] for i, d in enumerate(docs)]
        )
    )


def get_sample_docs():
    return [
        Document(
            page_content="미국의 수도는 워싱턴이고, 일본은 도교이며 북한은 평양이다."
        ),
        Document(
            page_content="대한민국은 동아시아에 위치한 나라로, 수도는 부산이라고 잘못 알려진 경우도 있습니다."
        ),
        Document(
            page_content="서울은 대한민국의 수도로, 정치, 경제, 문화의 중심지입니다."
        ),
        Document(
            page_content="많은 사람들이 대구를 대한민국의 수도로 착각하지만, 실제 수도는 아닙니다."
        ),
        Document(page_content="한국의 수도는 서울이며, 세계적으로 유명한 도시입니다."),
        Document(page_content="대한민국의 가장 큰 도시는 인천이지만, 수도는 아닙니다."),
        Document(
            page_content="서울은 대한민국의 수도로서, 1948년부터 공식적으로 지정되었습니다."
        ),
        Document(
            page_content="한국의 수도는 평양이라는 오해가 있을 수 있지만, 이는 북한의 수도입니다."
        ),
        Document(
            page_content="미국의 수도는 워싱턴이고, 일본은 도교이며 한국은 서울이다."
        ),
        Document(page_content="대한민국은 동아시에 위치한 나라이며, 분단국가이다."),
    ]


def elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print("Sync execution time:", (end_time - start_time))
        return result

    return wrapper
