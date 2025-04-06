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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.tools import tool

# 加载环境变量
dotenv.load_dotenv()

# 设置项目名称
os.environ["LANGSMITH_PROJECT"] = os.path.basename(__file__)

# 创建内存检查点保存器
memory = MemorySaver()


class State(TypedDict):
    """定义状态类型，包含消息列表"""
    messages: Annotated[list, add_messages]

# 创建图构建器
graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    """请求人工协助的工具函数"""
    human_response = interrupt({"query": query})
    return human_response["data"]

# 初始化搜索工具
tool = TavilySearchResults(max_results=1)
# 组合工具列表，包含搜索和人工协助
tools = [tool, human_assistance]

# 初始化 LLM
llm = ChatDeepSeek(model="deepseek-coder")
# 将工具绑定到 LLM
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State) -> State:
    """聊天机器人节点函数"""
    message = llm_with_tools.invoke(state["messages"])
    # 由于我们在工具执行期间会中断，
    # 我们禁用并行工具调用以避免在恢复时重复任何工具调用
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


# 添加节点和边
graph_builder.add_node("chatbot", chatbot)

# 创建工具节点
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# 添加条件边
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# 添加边连接
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# 编译图，使用内存检查点
graph = graph_builder.compile(checkpointer=memory)


# 测试用户输入
user_input = "我需要一些关于构建AI代理的专家指导。你能帮我请求协助吗？"
# 配置线程ID
config = {"configurable": {"thread_id": "1"}}

# 运行图并处理事件
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# 模拟人工响应
human_response = (
    "我们专家在这里为您提供帮助！我们建议您查看 LangGraph 来构建您的代理。"
    " 它比简单的自主代理更可靠和可扩展。"
)

# 创建恢复命令
human_command = Command(resume={"data": human_response})

# 继续处理人工响应
events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()