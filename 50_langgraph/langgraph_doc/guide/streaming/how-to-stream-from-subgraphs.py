from typing import Optional, Annotated

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# 로그 구조
class Logs(TypedDict):
    id: str
    question: str
    answer: str
    grade: Optional[int]
    feedback: Optional[str]


# 커스텀 리듀서 정의
def add_logs(left: list[Logs], right: list[Logs]) -> list[Logs]:
    if not left:
        left = []

    if not right:
        right = []

    logs = left.copy()
    left_id_to_idx = {log["id"]: idx for idx, log in enumerate(logs)}
    # 새로운 로그가 상태에 이미 있는 경우 업데이트하고 그렇지 않으면 추가합니다.
    for log in right:
        idx = left_id_to_idx.get(log["id"])
        if idx is not None:
            logs[idx] = log
        else:
            logs.append(log)
    return logs


# 실패 분석 서브그래프 상태
class FailureAnalysisState(TypedDict):
    # 부모 그래프와 공유되는 키 (EntryGraphState)
    logs: Annotated[list[Logs], add_logs]
    failure_report: str
    # 서브그래프 키
    failures: list[Logs]


def get_failures(state: FailureAnalysisState):
    failures = [log for log in state["logs"] if log["grade"] == 0]
    return {"failures": failures}


def generate_summary(state: FailureAnalysisState):
    failures = state["failures"]
    # 여기에 커스텀 요약 로직을 구현할 수 있다.
    failure_ids = [log["id"] for log in failures]
    fa_summary = f"Poor quality of retrieval for document IDs: {', '.join(failure_ids)}"
    return {"failure_report": fa_summary}


fa_builder = StateGraph(FailureAnalysisState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)


# 요약 서브그래프
class QuestionSummarizationState(TypedDict):
    # 부모 그래프와 공유되는 키 (EntryGraphState)
    summary_report: str
    logs: Annotated[list[Logs], add_logs]
    # 서브그래프 키
    summary: str


def generate_summary(state: QuestionSummarizationState):
    docs = state["logs"]
    # 여기에 커스텀 요약 로직을 구현할 수 있다.
    # summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    summary = "ChatOllama와 Chroma 벡터 저장소 사용에 대한 질문."
    return {"summary": summary}


def send_to_slack(state: QuestionSummarizationState):
    summary = state["summary"]
    # 여기에 커스텀 로직을 구현할 수 있다. 예를 들어, 이전 단계에서 생성된 요약을 Slack으로 전송할 수 있다.
    return {"summary_report": summary}


qs_builder = StateGraph(QuestionSummarizationState)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)


class EntryGraphState(TypedDict):
    raw_logs: Annotated[list[Logs], add_logs]
    logs: Annotated[list[Logs], add_logs]  # 서브그래프에서 사용된다
    failure_report: str  # FA 서브그래프에서 생성된다.
    summary_report: str  # QS 서브그래프에서 생성된다.


def select_logs(state):
    return {"logs": [log for log in state["raw_logs"] if "grade" in log]}


entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("select_logs", select_logs)
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())

entry_builder.add_edge(START, "select_logs")
entry_builder.add_edge("select_logs", "failure_analysis")
entry_builder.add_edge("select_logs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

graph = entry_builder.compile()

from IPython.display import Image, display

# 중첩 그래프의 내부 구조를 보여주기 위해 xray를 1로 설정한다.
display(
    Image(
        graph.get_graph(xray=1).draw_mermaid_png(
            output_file_path="how-to-stream-from-subgraphs.png"
        )
    )
)

# Dummy logs
dummy_logs = [
    Logs(
        id="1",
        question="ChatOllama를 어떻게 import 할 수 있어?",
        grade=1,
        answer="ChatOllama를 import 하기 위해서, 다음을 사용해: 'from langchain_community.chat_models import ChatOllama.'",
    ),
    Logs(
        id="2",
        question="Chroma vector store를 어떻게 사용할 수 있어?",
        answer="Chroma를 사용하기 위해 다음을 정의해: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
        grade=0,
        feedback="일반적으로 검색된 문서는 벡터 저장소에 대한 내용이고, 특별히 Chroma에 대한 것은 아니다",
    ),
    Logs(
        id="3",
        question="langgraph에서 react agent를 어떻게 만들 수 있어?",
        answer="from langgraph.prebuilt import create_react_agent",
    ),
]

input = {"raw_logs": dummy_logs}

for chunk in graph.stream(input, stream_mode="updates"):
    node_name = list(chunk.keys())[0]
    print(f"---------- Update from node {node_name} ---------")
    print(chunk[node_name])


# Format the namespace slightly nicer
def format_namespace(namespace):
    return (
        namespace[-1].split(":")[0] + " subgraph"
        if len(namespace) > 0
        else "parent graph"
    )


for namespace, chunk in graph.stream(input, stream_mode="updates", subgraphs=True):
    node_name = list(chunk.keys())[0]
    print(
        f"---------- Update from node {node_name} in {format_namespace(namespace)} ---------"
    )
    print(chunk[node_name])
