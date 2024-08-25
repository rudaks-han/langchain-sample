import requests


def get_stream(query: str):
    s = requests.Session()
    with s.post("http://localhost:8001/chat", stream=True, json={"text": query}) as r:
        print("response...")
        for line in r.iter_content():
            print(line.decode("utf-8"), end="")

        print("done")


get_stream("tell me a short story")
