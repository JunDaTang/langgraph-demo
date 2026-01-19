"""
LangGraph 短期记忆演示

该演示展示了如何使用短期记忆（线程级持久性）使智能体能够跟踪多轮对话。
"""

from typing import Annotated
from typing_extensions import TypedDict
import operator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

# 定义状态，包含消息历史
class ChatState(TypedDict):
  """聊天状态定义"""
  messages: Annotated[list, operator.add]
  user_name: str

def greeting_node(state: ChatState) -> dict:
  """
  问候节点
  
  Args:
    state: 当前状态
    
  Returns:
    dict: 更新后的状态
  """
  print("执行节点: greeting_node")
  
  user_name = state.get("user_name", "访客")
  greeting_message = f"你好，{user_name}！我是你的AI助手。"
  
  return {
    "messages": [("assistant", greeting_message)]
  }

def respond_node(state: ChatState) -> dict:
  """
  回应节点
  
  Args:
    state: 当前状态
    
  Returns:
    dict: 更新后的状态
  """
  print("执行节点: respond_node")
  
  # 获取最新的用户消息
  user_messages = [msg for msg in state["messages"] if msg[0] == "user"]
  if user_messages:
    latest_user_message = user_messages[-1][1]
    user_name = state.get("user_name", "访客")
    
    # 根据用户消息生成回应
    if "你好" in latest_user_message or "hello" in latest_user_message.lower():
      response = f"你好，{user_name}！有什么我可以帮助你的吗？"
    elif "天气" in latest_user_message:
      response = f"抱歉，{user_name}，我无法获取实时天气信息。"
    elif "名字" in latest_user_message or "我是" in latest_user_message:
      response = f"我知道你叫{user_name}，很高兴认识你！"
    else:
      response = f"我理解你说的，{user_name}。能告诉我更多吗？"
  else:
    response = "我没有看到你的消息，请再说一遍。"
  
  return {
    "messages": [("assistant", response)]
  }

def main():
  """主函数 - 演示短期记忆功能"""
  print("=== LangGraph 短期记忆演示 ===\n")
  
  # 创建内存存储器（短期记忆）
  memory = InMemorySaver()
  
  # 构建图
  builder = StateGraph(ChatState)
  builder.add_node("greeting", greeting_node)
  builder.add_node("respond", respond_node)
  
  builder.add_edge(START, "greeting")
  builder.add_edge("greeting", "respond")
  builder.add_edge("respond", END)
  
  # 编译图并使用内存存储
  graph = builder.compile(checkpointer=memory)
  
  # 配置线程ID用于存储状态
  config = {"configurable": {"thread_id": "chat_1"}}
  
  # 第一轮对话
  print("1. 第一轮对话:")
  result1 = graph.invoke({
    "messages": [("user", "你好！我叫张三")],
    "user_name": "张三"
  }, config)
  
  print("对话历史:")
  for role, message in result1["messages"]:
    print(f" {role}: {message}")
  print()
  
  # 查看存储的状态
  print("2. 检查存储的状态:")
  saved_state = graph.get_state(config)
  print("保存的对话历史:")
  for role, message in saved_state.values["messages"]:
    print(f" {role}: {message}")
  print()
  
  # 第二轮对话（继续之前的对话）
  print("3. 第二轮对话（继续之前的对话）:")
  result2 = graph.invoke({
    "messages": [("user", "今天天气怎么样？")],
    "user_name": "张三"
  }, config)
  
  print("对话历史:")
  for role, message in result2["messages"]:
    print(f" {role}: {message}")
  print()
  
  # 第三轮对话
  print("4. 第三轮对话:")
  result3 = graph.invoke({
    "messages": [("user", "你能记住我的名字吗？")],
    "user_name": "张三"
  }, config)
  
  print("对话历史:")
  for role, message in result3["messages"]:
    print(f" {role}: {message}")
  print()
  
  # 查看最终状态
  print("5. 最终状态:")
  final_state = graph.get_state(config)
  print("完整的对话历史:")
  for role, message in final_state.values["messages"]:
    print(f" {role}: {message}")
  print()
  
  print("=== 演示完成 ===")

if __name__ == "__main__":
  main()



# === LangGraph 短期记忆演示 ===

# 1. 第一轮对话:
# 执行节点: greeting_node
# 执行节点: respond_node
# 对话历史:
#  user: 你好！我叫张三
#  assistant: 你好，张三！我是你的AI助手。
#  assistant: 你好，张三！有什么我可以帮助你的吗？

# 2. 检查存储的状态:
# 保存的对话历史:
#  user: 你好！我叫张三
#  assistant: 你好，张三！我是你的AI助手。
#  assistant: 你好，张三！有什么我可以帮助你的吗？

# 3. 第二轮对话（继续之前的对话）:
# 执行节点: greeting_node
# 执行节点: respond_node
# 对话历史:
#  user: 你好！我叫张三
#  assistant: 你好，张三！我是你的AI助手。
#  assistant: 你好，张三！有什么我可以帮助你的吗？
#  user: 今天天气怎么样？
#  assistant: 你好，张三！我是你的AI助手。
#  assistant: 抱歉，张三，我无法获取实时天气信息。

# 4. 第三轮对话:
# 执行节点: greeting_node
# 执行节点: respond_node
# 对话历史:
#  user: 你好！我叫张三
#  assistant: 你好，张三！我是你的AI助手。
#  assistant: 你好，张三！有什么我可以帮助你的吗？
#  user: 今天天气怎么样？
#  assistant: 你好，张三！我是你的AI助手。
#  assistant: 抱歉，张三，我无法获取实时天气信息。
#  user: 你能记住我的名字吗？
#  assistant: 你好，张三！我是你的AI助手。
#  assistant: 我知道你叫张三，很高兴认识你！

# 5. 最终状态:
# 完整的对话历史:
#  user: 你好！我叫张三
#  assistant: 你好，张三！我是你的AI助手。
#  assistant: 你好，张三！有什么我可以帮助你的吗？
#  user: 今天天气怎么样？
#  assistant: 你好，张三！我是你的AI助手。
#  assistant: 抱歉，张三，我无法获取实时天气信息。
#  user: 你能记住我的名字吗？
#  assistant: 你好，张三！我是你的AI助手。
#  assistant: 我知道你叫张三，很高兴认识你！

# === 演示完成 ===