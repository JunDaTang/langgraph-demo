from typing import Annotated
from langchain_deepseek import ChatDeepSeek
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command, interrupt

# 定义自定义状态类型，包含消息列表、姓名和生日
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 消息历史
    name: str                               # 用户姓名
    birthday: str                           # 用户生日


@tool
# 注意：由于我们要为状态更新生成 ToolMessage，
# 通常需要对应工具调用的 ID。
# 我们可以使用 LangChain 的 InjectedToolCallId 来指示这个参数
# 不应该在工具的模式中显示给模型
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """请求人工协助验证用户信息"""
    human_response = interrupt(
        {
            "question": "这些信息是否正确？",
            "name": name,
            "birthday": birthday,
        },
    )
    # 如果信息正确，按原样更新状态
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "正确"
    # 否则，从人工审核者接收更正的信息
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"已更正: {human_response}"

    # 这次我们在工具内部使用 ToolMessage 显式更新状态
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # 我们在工具中返回一个 Command 对象来更新状态
    return Command(update=state_update)


from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# 初始化搜索工具和人工协助工具
tool = TavilySearchResults(max_results=5)
tools = [tool, human_assistance]

# 初始化 LLM 并绑定工具
llm = ChatDeepSeek(model="deepseek-coder")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    """聊天机器人节点函数"""
    message = llm_with_tools.invoke(state["messages"])
    # 确保每次只调用一个工具
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


# 构建图结构
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

# 创建工具节点
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# 添加条件边和连接
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# 创建内存检查点并编译图
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# 测试用户输入
user_input = (
    "你能查找一下 LangGraph 是什么时候发布的吗？"
    " 当你找到答案后，使用 human_assistance 工具进行审核。"
)
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



        

human_command = Command(
    resume={
        "name": "LangGraph",
        "birthday": "Jan 17, 2024",
    },
)

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()