"""
LangGraph SQLite 长期记忆演示

该演示展示了如何在生产环境中使用 SQLite 数据库作为长期记忆存储。
参考 PostgreSQL 的正确用法，使用上下文管理器来避免事务冲突。
"""

import sqlite3
import uuid
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore 
from langgraph.store.base import BaseStore

# 定义状态（使用 MessagesState 简化代码）
class ChatState(MessagesState):
  """聊天状态定义 - 继承自 MessagesState"""
  pass

def call_model(
  state: ChatState,
  config: RunnableConfig,
  *,
  store: BaseStore, 
):
  """
  调用模型的节点函数
  
  Args:
    state: 当前状态
    config: 配置信息
    store: 存储对象
    
  Returns:
    dict: 更新后的状态
  """
  print("执行节点: call_model")
  
  # 从配置中获取用户ID
  user_id = config["configurable"]["user_id"]
  namespace = ("memories", user_id)
  
  # 从存储中搜索相关记忆
  try:
    memories = store.search(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])
    print(f"检索到的记忆: {info}")
  except Exception as e:
    print(f"检索记忆时出错: {e}")
    info = ""
  
  system_msg = f"你是一个有帮助的助手，正在与用户交谈。用户信息: {info}" if info else "你是一个有帮助的助手。"
  print(f"系统消息: {system_msg}")
  
  # 检查用户是否要求记住某些信息
  last_message = state["messages"][-1]
  if "记住" in last_message.content.lower() or "remember" in last_message.content.lower():
    # 提取需要记住的信息（这里简化处理）
    memory = "用户的名字是张三" if "张三" in last_message.content else "用户要求记住一些信息"
    try:
      store.put(namespace, str(uuid.uuid4()), {"data": memory})
      print(f"已存储记忆: {memory}")
    except Exception as e:
      print(f"存储记忆时出错: {e}")
  
  # 生成回复（这里使用模拟回复代替实际模型调用）
  user_message = last_message.content
  if "你好" in user_message or "hello" in user_message.lower():
    response = "你好！我是AI助手。有什么我可以帮助你的吗？"
  elif "记住" in user_message.lower() or "remember" in user_message.lower():
    response = "好的，我已经记住了你说的信息。"
  elif "名字" in user_message or "name" in user_message.lower():
    if info:
      response = f"根据我的记忆，你的名字是张三。"
    else:
      response = "我还不知道你的名字，能告诉我吗？"
  else:
    response = "我理解你的问题。让我来帮助你解答。"
  
  print(f"生成的回复: {response}")
  return {"messages": [AIMessage(content=response)]}

def main():
  """主函数 - 演示 SQLite 长期记忆功能"""
  print("=== LangGraph SQLite 长期记忆演示 ===\n")
  
  DB_PATH = "sgg/chapter04-高级特性/sqlite_data/long_term_memory.db"
  
  # 使用上下文管理器确保正确初始化和清理资源
  with (
    SqliteStore.from_conn_string(DB_PATH) as store,
    SqliteSaver.from_conn_string(DB_PATH) as checkpointer,
  ):
    # 构建图
    builder = StateGraph(ChatState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    # 编译图，同时使用检查点和存储
    graph = builder.compile(
      checkpointer=checkpointer,
      store=store, 
    )

    # 第一次对话 - 要求记住信息
    print("1. 第一次对话 - 要求记住信息:")
    config1 = {
      "configurable": {
        "thread_id": "1", 
        "user_id": "user_123", 
      }
    }
    
    for chunk in graph.stream(
      {"messages": [HumanMessage(content="你好！请记住：我的名字是张三")]},
      config1, 
      stream_mode="values",
    ):
      if chunk["messages"]:
        last_message = chunk["messages"][-1]
        if hasattr(last_message, 'content'):
          print(f" {type(last_message).__name__}: {last_message.content}")
        else:
          print(f" {type(last_message).__name__}: {last_message}")
    print()

    # 第二次对话 - 查询记忆
    print("2. 第二次对话 - 查询记忆:")
    config2 = {
      "configurable": {
        "thread_id": "2", 
        "user_id": "user_123",
      }
    }

    for chunk in graph.stream(
      {"messages": [HumanMessage(content="我的名字是什么？")]},
      config2, 
      stream_mode="values",
    ):
      if chunk["messages"]:
        last_message = chunk["messages"][-1]
        if hasattr(last_message, 'content'):
          print(f" {type(last_message).__name__}: {last_message.content}")
        else:
          print(f" {type(last_message).__name__}: {last_message}")
    print()
    
    # 第三次对话 - 不同用户
    print("3. 第三次对话 - 不同用户:")
    config3 = {
      "configurable": {
        "thread_id": "3", 
        "user_id": "user_456", # 不同的用户ID
      }
    }

    for chunk in graph.stream(
      {"messages": [HumanMessage(content="我的名字是什么？")]},
      config3, 
      stream_mode="values",
    ):
      if chunk["messages"]:
        last_message = chunk["messages"][-1]
        if hasattr(last_message, 'content'):
          print(f" {type(last_message).__name__}: {last_message.content}")
        else:
          print(f" {type(last_message).__name__}: {last_message}")
    print()
  
  print("=== 演示完成 ===")

if __name__ == "__main__":
  main()


# === LangGraph SQLite 长期记忆演示 ===

# 1. 第一次对话 - 要求记住信息:
#  HumanMessage: 你好！请记住：我的名字是张三
# 执行节点: call_model
# 检索到的记忆: 
# 系统消息: 你是一个有帮助的助手。
# 已存储记忆: 用户的名字是张三
# 生成的回复: 你好！我是AI助手。有什么我可以帮助你的吗？
#  AIMessage: 你好！我是AI助手。有什么我可以帮助你的吗？

# 2. 第二次对话 - 查询记忆:
#  HumanMessage: 我的名字是什么？
# 执行节点: call_model
# 检索到的记忆: 用户的名字是张三
# 系统消息: 你是一个有帮助的助手，正在与用户交谈。用户信息: 用户的名字是张三
# 生成的回复: 根据我的记忆，你的名字是张三。
#  AIMessage: 根据我的记忆，你的名字是张三。

# 3. 第三次对话 - 不同用户:
#  HumanMessage: 我的名字是什么？
# 执行节点: call_model
# 检索到的记忆:
# 系统消息: 你是一个有帮助的助手。
# 生成的回复: 我还不知道你的名字，能告诉我吗？
#  AIMessage: 我还不知道你的名字，能告诉我吗？

# === 演示完成 ===