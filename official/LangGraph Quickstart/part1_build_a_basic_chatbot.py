
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_deepseek import ChatDeepSeek
from IPython.display import Image, display
import dotenv
import os

dotenv.load_dotenv()


os.environ["LANGSMITH_PROJECT"] = os.path.basename(__file__)

class State(TypedDict):
    # messages的类型是"list"。注解中的`add_messages`函数定义了如何更新这个状态键
    # (在这种情况下,它是将消息追加到列表中,而不是覆盖它们)
    messages: Annotated[list, add_messages]



graph_builder = StateGraph(State)

# llm = ChatDeepSeek(model="deepseek-coder",api_key=os.getenv("DEEPSEEK_API_KEY"),api_base=os.getenv("DEEPSEEK_API_URL"))
llm = ChatDeepSeek(model="deepseek-coder")

def chatbot(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

# for event in graph.stream({"messages": [{"role": "user", "content": '你好'}]}):
#     for value in event.values():
#         print("Assistant:", value["messages"][-1].content)

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print(e.traceback())
        # fallback if input() is not available
        print("Goodbye!")
        break


