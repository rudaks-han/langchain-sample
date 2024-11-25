import operator
from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

subjects_prompt = """Generate a comma separated list of between 2 and 5 examples related to: {topic}."""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one.

{jokes}"""


class Subjects(BaseModel):
    subjects: list[str]


class Joke(BaseModel):
    joke: str


class BestJoke(BaseModel):
    id: int = Field(description="Index of the best joke, starting with 0", ge=0)


model = ChatOpenAI(model="gpt-4o-mini")


class OverallState(TypedDict):
    topic: str
    subjects: list
    # 개별 노드에서 생성한 모든 농담을 하나의 목록으로 결합하려고 하기 때문에 여기서 operator.add를 사용한다.
    # 이것은 본질적으로 "reduce" 부분이다.
    jokes: Annotated[list, operator.add]
    best_selected_joke: str


# 농담을 생성하기 위해 모든 주제를 "매핑"할 노드의 상태가 될 것이다.
class JokeState(TypedDict):
    subject: str


# 농담의 주제를 생성하는 데 사용할 함수이다.
def generate_topics(state: OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}


# 주제에 대한 농담을 생성하는 데 사용할 함수이다.
def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}


# 여기서 우리는 생성된 주제에 대해 매핑할 로직을 정의한다.
# 우리는 그래프의 엣지로 이것을 사용할 것이다.
def continue_to_jokes(state: OverallState):
    # 'Send' 객체의 리스트를 반환할 것이다.
    # 각 'Send' 객체는 그래프의 노드 이름과 해당 노드로 보낼 상태로 구성된다.
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


# 최적의 농담을 판단할 것이다.
def best_joke(state: OverallState):
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}


graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)
graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)
app = graph.compile()

from IPython.display import Image

Image(app.get_graph().draw_mermaid_png(output_file_path="./map-reduce-branches.png"))

for s in app.stream({"topic": "동물"}):
    print(s)
