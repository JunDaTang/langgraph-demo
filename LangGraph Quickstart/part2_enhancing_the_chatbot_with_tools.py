from IPython.display import Image, display
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_deepseek import ChatDeepSeek
import dotenv
import os
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display

dotenv.load_dotenv()

os.environ["LANGSMITH_PROJECT"] = os.path.basename(__file__)


class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=1)
tools = [tool]
# print(tool.invoke("今天广东天气如何"))

# llm = ChatDeepSeek(model="deepseek-coder",api_key=os.getenv("DEEPSEEK_API_KEY"),api_base=os.getenv("DEEPSEEK_API_URL"))
# llm_with_tools = ChatDeepSeek(model="deepseek-coder",tools=tools)
llm = ChatDeepSeek(model="deepseek-coder")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State) -> State:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
# graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()



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


