from IPython.display import Image, display
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_deepseek import ChatDeepSeek
import dotenv
import os
# from langgraph.prebuilt import ToolNode, tools_condition
from part2.basic_tool_node import BasicToolNode
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

# tool_node = ToolNode(tools=tools)
# graph_builder.add_node("tools", tool_node)
tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# graph_builder.add_conditional_edges(
#     "chatbot",
#     tools_condition,
# )
def route_tools(
    state: State,
):
    """
    在条件边中使用，用于路由到工具节点（如果最后一条消息包含工具调用）。
    否则，路由到结束节点。
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# `tools_condition` 函数在聊天机器人请求使用工具时返回 "tools"，
# 在直接回复时返回 "END"。这个条件路由定义了主要的代理循环。
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # 下面的字典让你告诉图如何解释条件的输出作为特定节点
    # 它默认为恒等函数，但如果你
    # 想使用一个名为 "tools" 以外的节点，
    # 你可以更新字典的值
    # 例如: "tools": "my_tools"
    {"tools": "tools", END: END},
)
# 每次调用工具后，我们返回到聊天机器人来决定下一步
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # 这需要一些额外的依赖项，是可选的
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
        # 如果 input() 不可用时的后备方案
        print("Goodbye!")
        break


