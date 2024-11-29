from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display


class State(TypedDict):
    input: str


def step_1(state):
    print("---Step 1---")
    pass


def step_2(state):
    print("---Step 2---")
    pass


def step_3(state):
    print("---Step 3---")
    pass


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# 메모리 설정
memory = MemorySaver()

graph = builder.compile(checkpointer=memory, interrupt_before=["step_3"])

# View
display(
    Image(
        graph.get_graph().draw_mermaid_png(output_file_path="how-to-add-breakpoint.png")
    )
)

initial_input = {"input": "안녕~"}

thread = {"configurable": {"thread_id": "1"}}

# 첫 번째 중단까지 그래프 실행
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)

try:
    user_approval = input("3단계로 가시겠습니까? (yes/no): ")
except:
    user_approval = "yes"

if user_approval.lower() == "yes":
    # input에 None 입력 시 이전 상태에서 그래프 실행 계속
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
else:
    print("Operation cancelled by user.")
