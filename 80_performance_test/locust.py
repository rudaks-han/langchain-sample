from locust import HttpUser, task


class HelloWorldUser(HttpUser):
    host = "http://localhost:8000"  # 테스트 대상 호스트 주소 지정

    @task
    def request(self):
        # url = "/request/sync"
        # url = "/request/async"
        # url = "/request/openai/sync"
        # url = "/request/openai/async"
        # url = "/request/langchain/sync"
        # url = "/request/langchain/async"
        # url = "/request/retrieve/sync"
        # url = "/request/retrieve/async"
        # url = "/request/rag/sync"
        # url = "/request/rag/async"
        # url = "/request/retrieval_chain/sync"
        # url = "/request/rag/vector"
        url = "/request/rag/ensemble"
        self.client.get(url)
