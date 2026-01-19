"""
基于LangGraph 1.0函数式API的长期记忆示例

本示例演示了如何在LangGraph函数式API中实现长期记忆功能，
展示了跨不同线程ID存储信息的能力。

主要特性：
1. 使用共享存储实现长期记忆
2. 跨会话保持用户信息
3. 使用@entrypoint装饰器定义工作流入口点
4. 演示多线程ID间的信息共享
"""

from typing import Dict, List, Optional
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
import time
import json

# 模拟长期记忆存储（在实际应用中可能是数据库或外部存储）
class LongTermMemoryStore:
  """长期记忆存储类"""
  def __init__(self):
    self.storage: Dict[str, Dict] = {}
  
  def save_user_info(self, user_id: str, info: Dict) -> None:
    """
    保存用户信息到长期记忆
    
    参数:
      user_id: 用户ID
      info: 用户信息字典
    """
    print(f"保存用户 {user_id} 的信息到长期记忆: {info}")
    if user_id not in self.storage:
      self.storage[user_id] = {}
    self.storage[user_id].update(info)
  
  def get_user_info(self, user_id: str) -> Dict:
    """
    从长期记忆获取用户信息
    
    参数:
      user_id: 用户ID
      
    返回:
      用户信息字典
    """
    info = self.storage.get(user_id, {})
    print(f"从长期记忆获取用户 {user_id} 的信息: {info}")
    return info
  
  def list_users(self) -> List[str]:
    """
    列出所有用户
    
    返回:
      用户ID列表
    """
    return list(self.storage.keys())

# 全局长期记忆存储实例
long_term_memory = LongTermMemoryStore()

@task
def extract_user_info(message: str) -> Dict:
  """
  从消息中提取用户信息的任务
  
  参数:
    message: 用户消息
    
  返回:
    提取的用户信息字典
  """
  print(f"从消息中提取用户信息: {message}")
  
  user_info = {}
  
  # 简单的信息提取逻辑
  if "我叫" in message or "我是" in message:
    if "我叫" in message:
      name_start = message.find("我叫") + 2
    else:
      name_start = message.find("我是") + 2
      
    # 提取到下一个标点符号或字符串结尾
    name_end = len(message)
    for i in range(name_start, len(message)):
      if message[i] in [",", "，", ".", "。", "!", "！", "?", "？"]:
        name_end = i
        break
    name = message[name_start:name_end].strip()
    if name: # 只有非空名称才保存
      user_info["name"] = name
  
  if "岁" in message:
    # 查找年龄数字
    age_pos = message.find("岁")
    # 向前查找数字
    age_str = ""
    for i in range(age_pos - 1, -1, -1):
      if message[i].isdigit():
        age_str = message[i] + age_str
      else:
        break
    if age_str and age_str.isdigit():
      user_info["age"] = int(age_str)
  
  if "来自" in message:
    location_start = message.find("来自") + 2
    # 提取到下一个标点符号或字符串结尾
    location_end = len(message)
    for i in range(location_start, len(message)):
      if message[i] in [",", "，", ".", "。", "!", "！", "?", "？"]:
        location_end = i
        break
    location = message[location_start:location_end].strip()
    if location: # 只有非空位置才保存
      user_info["location"] = location
  
  print(f"提取到的用户信息: {user_info}")
  return user_info

@task
def generate_response(user_id: str, message: str, user_info: Dict) -> str:
  """
  生成回复的任务
  
  参数:
    user_id: 用户ID
    message: 用户消息
    user_info: 用户信息
    
  返回:
    生成的回复
  """
  print(f"为用户 {user_id} 生成回复，消息: {message}，用户信息: {user_info}")
  
  # 根据用户信息生成个性化回复
  if "你好" in message or "hello" in message.lower():
    if user_info.get("name"):
      response = f"你好，{user_info['name']}！很高兴再次见到你。"
    else:
      response = "你好！我是AI助手。能告诉我你的名字吗？"
  elif "我叫" in message or "我是" in message:
    name = user_info.get("name", "朋友")
    response = f"很高兴认识你，{name}！有什么我可以帮助你的吗？"
  elif "再见" in message or "bye" in message.lower():
    name = user_info.get("name", "朋友")
    response = f"再见，{name}！期待下次与你交流。"
  else:
    # 基于用户信息的个性化回复
    info_parts = []
    if user_info.get("name"):
      info_parts.append(f"名字是{user_info['name']}")
    if user_info.get("age"):
      info_parts.append(f"年龄是{user_info['age']}岁")
    if user_info.get("location"):
      info_parts.append(f"来自{user_info['location']}")
      
    if info_parts:
      info_summary = "，而且我知道你" + "，".join(info_parts)
      response = f"我理解你的问题。{info_summary}。让我来帮助你解答。"
    else:
      response = "我理解你的问题。让我来帮助你解答。"
  
  print(f"生成的回复: {response}")
  return response

# 检查点保存器
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def long_term_memory_chat(inputs: Dict, *, previous_context: Optional[Dict] = None) -> Dict:
  """
  具有长期记忆的聊天工作流
  
  参数:
    inputs: 包含user_id和message的字典
    previous_context: 之前的上下文信息（来自检查点）
    
  返回:
    包含回复和更新后用户信息的字典
  """
  user_id = inputs["user_id"]
  message = inputs["message"]
  
  print(f"\n=== 启动长期记忆聊天工作流 ===")
  print(f"用户ID: {user_id}")
  print(f"消息: {message}")
  print(f"之前的上下文: {previous_context}")
  
  # 从长期记忆获取用户信息
  stored_user_info = long_term_memory.get_user_info(user_id)
  
  # 合并存储的用户信息和之前的上下文
  current_user_info = {}
  if stored_user_info:
    current_user_info.update(stored_user_info)
  if previous_context:
    current_user_info.update(previous_context)
  
  print(f"当前用户信息: {current_user_info}")
  
  # 提取用户信息
  extracted_info = extract_user_info(message).result()
  
  # 更新用户信息
  if extracted_info:
    current_user_info.update(extracted_info)
    # 保存到长期记忆
    long_term_memory.save_user_info(user_id, current_user_info)
  
  # 生成回复
  response = generate_response(user_id, message, current_user_info).result()
  
  # 返回回复和更新后的用户信息
  result = {
    "user_id": user_id,
    "message": message,
    "response": response,
    "user_info": current_user_info
  }
  
  print(f"工作流结果: {result}")
  
  # 保存用户信息到检查点（短期记忆）
  return entrypoint.final(value=result, save=current_user_info)

def conversation_session_1():
  """第一次对话会话"""
  print("=== 第一次对话会话 ===")
  
  user_id = "user_001"
  config = {"configurable": {"thread_id": "session_1"}}
  
  # 第一条消息
  print("\n--- 第一条消息 ---")
  message1 = "你好，我叫张三，来自北京。"
  result1 = long_term_memory_chat.invoke({"user_id": user_id, "message": message1}, config=config)
  print(f"用户: {message1}")
  print(f"AI: {result1['response']}")
  print(f"当前用户信息: {result1['user_info']}")
  
  # 第二条消息
  print("\n--- 第二条消息 ---")
  message2 = "我今年25岁了。"
  result2 = long_term_memory_chat.invoke({"user_id": user_id, "message": message2}, config=config)
  print(f"用户: {message2}")
  print(f"AI: {result2['response']}")
  print(f"当前用户信息: {result2['user_info']}")

def conversation_session_2():
  """第二次对话会话（不同的线程ID）"""
  print("\n\n=== 第二次对话会话（不同线程ID） ===")
  
  user_id = "user_001" # 相同的用户ID
  config = {"configurable": {"thread_id": "session_2"}} # 不同的线程ID
  
  # 第一条消息
  print("\n--- 第一条消息 ---")
  message1 = "你好！"
  result1 = long_term_memory_chat.invoke({"user_id": user_id, "message": message1}, config=config)
  print(f"用户: {message1}")
  print(f"AI: {result1['response']}")
  print(f"当前用户信息: {result1['user_info']}")
  
  # 第二条消息
  print("\n--- 第二条消息 ---")
  message2 = "再见！"
  result2 = long_term_memory_chat.invoke({"user_id": user_id, "message": message2}, config=config)
  print(f"用户: {message2}")
  print(f"AI: {result2['response']}")
  print(f"当前用户信息: {result2['user_info']}")

def conversation_session_3():
  """第三次对话会话（新用户）"""
  print("\n\n=== 第三次对话会话（新用户） ===")
  
  user_id = "user_002" # 新的用户ID
  config = {"configurable": {"thread_id": "session_3"}}
  
  # 第一条消息
  print("\n--- 第一条消息 ---")
  message1 = "你好，我是李四，来自上海。"
  result1 = long_term_memory_chat.invoke({"user_id": user_id, "message": message1}, config=config)
  print(f"用户: {message1}")
  print(f"AI: {result1['response']}")
  print(f"当前用户信息: {result1['user_info']}")

def demonstrate_long_term_memory():
  """演示长期记忆功能"""
  print("=== 长期记忆功能演示 ===")
  
  print("\n当前长期记忆中的用户:")
  users = long_term_memory.list_users()
  for user_id in users:
    user_info = long_term_memory.get_user_info(user_id)
    print(f" {user_id}: {user_info}")

def multi_user_demo():
  """多用户演示"""
  print("\n\n=== 多用户演示 ===")
  
  # 用户1
  print("\n--- 用户1对话 ---")
  user1_id = "user_003"
  config1 = {"configurable": {"thread_id": "user1_session"}}
  message1 = "你好，我是王五，28岁。"
  result1 = long_term_memory_chat.invoke({"user_id": user1_id, "message": message1}, config=config1)
  print(f"用户1: {message1}")
  print(f"AI: {result1['response']}")
  
  # 用户2
  print("\n--- 用户2对话 ---")
  user2_id = "user_004"
  config2 = {"configurable": {"thread_id": "user2_session"}}
  message2 = "你好，我是赵六，来自深圳。"
  result2 = long_term_memory_chat.invoke({"user_id": user2_id, "message": message2}, config=config2)
  print(f"用户2: {message2}")
  print(f"AI: {result2['response']}")
  
  # 用户1再次对话
  print("\n--- 用户1再次对话 ---")
  message3 = "再见！"
  result3 = long_term_memory_chat.invoke({"user_id": user1_id, "message": message3}, config=config1)
  print(f"用户1: {message3}")
  print(f"AI: {result3['response']}")
  
  # 用户2再次对话
  print("\n--- 用户2再次对话 ---")
  message4 = "我今年30岁了。"
  result4 = long_term_memory_chat.invoke({"user_id": user2_id, "message": message4}, config=config2)
  print(f"用户2: {message4}")
  print(f"AI: {result4['response']}")

def check_memory_status():
  """检查内存状态"""
  print("\n\n=== 最终内存状态 ===")
  
  print("\n长期记忆存储内容:")
  users = long_term_memory.list_users()
  if users:
    for user_id in users:
      user_info = long_term_memory.get_user_info(user_id)
      print(f" {user_id}: {user_info}")
  else:
    print(" 没有存储的用户信息")
  
  print("\n检查点状态:")
  # 检查各个会话的状态
  sessions = ["session_1", "session_2", "session_3", "user1_session", "user2_session"]
  for session in sessions:
    try:
      config = {"configurable": {"thread_id": session}}
      state = long_term_memory_chat.get_state(config)
      if state.values:
        print(f" {session}: {state.values}")
      else:
        print(f" {session}: 无状态")
    except Exception as e:
      print(f" {session}: 无法获取状态 ({e})")

if __name__ == "__main__":
  # 执行演示
  conversation_session_1()
  conversation_session_2()
  conversation_session_3()
  demonstrate_long_term_memory()
  multi_user_demo()
  check_memory_status()