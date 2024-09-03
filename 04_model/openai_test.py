from os import environ

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=environ.get("OPENAI_API_KEY"))


def streaming():
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "한국의 수도는?"}],
        stream=True,
    )

    for chunk in stream:
        print("chunk", chunk.json())
        if chunk.choices[0].delta.content is not None:
            print(
                chunk.choices[0].delta.content,
            )


# streaming()


def not_streaming():
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "한국의 수도는?"}],
        stream=False,
    )

    print("completion", completion.json())
    print(completion.choices[0].message.content)


not_streaming()
# streaming()
