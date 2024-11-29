from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    input: str
    user_feedback: str


def step_1(state):
    print("---Step 1---")
    pass


def human_feedback(state):
    print("---human_feedback---")
    pass


def step_3(state):
    print("---Step 3---")
    pass


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])

display(
    Image(
        graph.get_graph().draw_mermaid_png(
            output_file_path="how-to-wait-for-user-input.png"
        )
    )
)

initial_input = {"input": "안녕~"}

thread = {"configurable": {"thread_id": "1"}}

for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)


try:
    user_input = input("피드백 주세요: ")
except:
    user_input = "go to step 3!"

# human_feedback 노드에 있는 것처럼 상태를 업데이트한다.
graph.update_state(thread, {"user_feedback": user_input}, as_node="human_feedback")

# 상태 확인
print("--State after update--")
print(graph.get_state(thread))

# human_feedback 다음에 3번 노드가 있는지 확인하기 위해 다음 노드를 확인한다.
print(graph.get_state(thread).next)

for event in graph.stream(None, thread, stream_mode="values"):
    print(event)

print(graph.get_state(thread).values)
