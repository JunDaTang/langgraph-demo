from typing import Annotated


from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# 定义状态类型，包含消息列表
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 消息历史记录


# 创建图构建器
graph_builder = StateGraph(State)


# 初始化搜索工具和 LLM
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatDeepSeek(model="deepseek-coder")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    """聊天机器人节点函数"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# 添加节点和边
graph_builder.add_node("chatbot", chatbot)

# 创建工具节点
tool_node = ToolNode(tools=[tool])
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


# 配置线程ID
config = {"configurable": {"thread_id": "1"}}

# 运行初始对话
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "我正在学习 LangGraph。"
                    " 你能帮我研究一下吗？"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


# 继续对话
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "是的，这很有帮助。"
                    " 也许我会用它来构建一个自主代理！"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# 遍历状态历史
to_replay = None
for state in graph.get_state_history(config):
    print("消息数量: ", len(state.values["messages"]), "下一步: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # 我们根据状态中的聊天消息数量选择特定状态
        to_replay = state
print(to_replay.next)
print(to_replay.config)


# `to_replay.config` 中的 `checkpoint_id` 对应我们已保存到检查点的状态
# 从选定的状态继续对话
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()