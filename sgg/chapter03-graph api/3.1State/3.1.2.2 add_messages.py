"""
LangGraph Reducer函数演示 - add_messages Reducer（消息列表专用）
"""

from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 2. add_messages Reducer（消息列表专用）
class AddMessagesState(TypedDict):
  messages: Annotated[List, add_messages]

def chat_node_1(state: AddMessagesState) -> dict:
  return {"messages": [("assistant", "Hello from node 1")]}

def chat_node_2(state: AddMessagesState) -> dict:
  return {"messages": [("assistant", "Hello from node 2")]}

def run_demo():
  print("2. add_messages Reducer（消息列表专用）演示:")
  builder = StateGraph(AddMessagesState)
  builder.add_node("chat1", chat_node_1)
  builder.add_node("chat2", chat_node_2)
  builder.add_edge(START, "chat1")
  builder.add_edge(START, "chat2") # 并行执行
  builder.add_edge("chat1", END)
  builder.add_edge("chat2", END)
  graph = builder.compile()
  
  result = graph.invoke({"messages": [("user", "Hi there!")]})
  print(f"初始状态: {{'messages': [('user', 'Hi there!')]}}")
  print(f"执行结果: {result}\n")

if __name__ == "__main__":
  run_demo()