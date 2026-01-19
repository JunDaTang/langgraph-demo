"""
基于LangGraph 1.0函数式API的短期记忆示例

本示例演示了如何在LangGraph函数式API中使用短期记忆功能，
展示了检查点机制和状态管理。

主要特性：
1. 使用InMemorySaver实现检查点存储
2. 使用@entrypoint装饰器定义带检查点的工作流入口点
3. 展示状态查看和历史记录功能
4. 演示线程状态的持久化和恢复
"""

from typing import List, Optional, Dict, Any
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import add_messages
import time

# 简单的累加器示例
@entrypoint(checkpointer=InMemorySaver())
def accumulate(n: int, *, previous: Optional[int] = None) -> entrypoint.final[int, int]:
  """
  累加器示例，展示如何解耦返回值和保存值
  
  参数:
    n: 当前输入数字
    previous: 之前的累加值
    
  返回:
    entrypoint.final对象，包含返回值和保存值
  """
  print(f"执行累加任务，输入: {n}")
  
  # 获取之前的值，如果没有则为0
  previous = previous or 0
  print(f"之前的累加值: {previous}")
  
  # 计算新的总和
  total = previous + n
  print(f"新的累加值: {total}")
  
  # 返回之前的值给调用者，但保存新的总和到检查点
  return entrypoint.final(value=previous, save=total)

def accumulate_demo():
  """累加器演示"""
  print("=== LangGraph 1.0 函数式API累加器演示 ===")
  
  # 配置工作流执行参数
  config = {"configurable": {"thread_id": "accumulate-thread"}}
  
  print("\n--- 第一次调用累加器 ---")
  result1 = accumulate.invoke(1, config=config)
  print(f"第一次调用结果: {result1}")
  
  print("\n--- 第二次调用累加器 ---")
  result2 = accumulate.invoke(2, config=config)
  print(f"第二次调用结果: {result2}")
  
  print("\n--- 第三次调用累加器 ---")
  result3 = accumulate.invoke(3, config=config)
  print(f"第三次调用结果: {result3}")
  
  print("\n--- 查看累加器线程状态 ---")
  # 查看当前线程状态
  current_state = accumulate.get_state(config)
  print(f"当前状态值: {current_state.values}")
  print(f"元数据: {current_state.metadata}")
  
  print("\n--- 查看累加器历史记录 ---")
  # 查看线程历史记录
  history = list(accumulate.get_state_history(config))
  print(f"历史记录条数: {len(history)}")
  
  for i, state in enumerate(history):
    print(f"\n历史状态 {i+1}:")
    print(f" 值: {state.values}")
    print(f" 创建时间: {state.created_at}")
    print(f" 步骤: {state.metadata.get('step', 'N/A')}")

# 状态管理示例
@entrypoint(checkpointer=InMemorySaver())
def state_manager(data: Dict[str, Any], *, previous: Optional[Dict[str, Any]] = None) -> entrypoint.final[Dict[str, Any], Dict[str, Any]]:
  """
  状态管理示例
  
  参数:
    data: 当前数据
    previous: 之前的状态
    
  返回:
    entrypoint.final对象，包含返回值和保存值
  """
  print(f"处理数据: {data}")
  
  # 如果有之前的状态，则合并
  if previous:
    print(f"之前状态: {previous}")
    # 合并状态
    new_state = {**previous, **data}
  else:
    new_state = data
  
  print(f"新状态: {new_state}")
  
  # 返回新状态给调用者，同时保存到检查点
  return entrypoint.final(value=new_state, save=new_state)

def state_management_demo():
  """状态管理演示"""
  print("\n\n=== 状态管理演示 ===")
  
  config = {"configurable": {"thread_id": "state-management-thread"}}
  
  # 第一次调用
  print("\n--- 第一次调用状态管理器 ---")
  data1 = {"name": "张三", "age": 25}
  result1 = state_manager.invoke(data1, config=config)
  print(f"第一次调用结果: {result1}")
  
  # 第二次调用
  print("\n--- 第二次调用状态管理器 ---")
  data2 = {"city": "北京", "occupation": "工程师"}
  result2 = state_manager.invoke(data2, config=config)
  print(f"第二次调用结果: {result2}")
  
  # 第三次调用
  print("\n--- 第三次调用状态管理器 ---")
  data3 = {"age": 26, "hobby": "编程"}
  result3 = state_manager.invoke(data3, config=config)
  print(f"第三次调用结果: {result3}")
  
  print("\n--- 查看状态管理线程状态 ---")
  current_state = state_manager.get_state(config)
  print(f"当前状态值: {current_state.values}")
  
  print("\n--- 查看状态管理历史记录 ---")
  history = list(state_manager.get_state_history(config))
  print(f"历史记录条数: {len(history)}")
  
  for i, state in enumerate(history):
    print(f"\n历史状态 {i+1}:")
    print(f" 值: {state.values}")
    print(f" 创建时间: {state.created_at}")
    print(f" 步骤: {state.metadata.get('step', 'N/A')}")

# 对话历史管理示例
@entrypoint(checkpointer=InMemorySaver())
def conversation_history(message: str, *, previous: Optional[List[str]] = None) -> entrypoint.final[str, List[str]]:
  """
  对话历史管理示例
  
  参数:
    message: 当前消息
    previous: 对话历史
    
  返回:
    entrypoint.final对象，包含返回值和保存值
  """
  print(f"收到消息: {message}")
  
  # 初始化历史记录
  history = previous or []
  
  print(f"当前历史记录: {history}")
  
  # 添加当前消息到历史记录
  updated_history = history + [message]
  
  # 生成回复（简单模拟）
  if len(updated_history) == 1:
    response = f"你好！我收到了你的消息: '{message}'"
  else:
    response = f"你之前说过: {', '.join(history[-2:])}。现在你说: '{message}'"
  
  print(f"生成回复: {response}")
  
  # 返回回复给调用者，保存更新后的历史记录
  return entrypoint.final(value=response, save=updated_history)

def conversation_demo():
  """对话历史演示"""
  print("\n\n=== 对话历史管理演示 ===")
  
  config = {"configurable": {"thread_id": "conversation-thread"}}
  
  # 第一轮对话
  print("\n--- 第一轮对话 ---")
  message1 = "你好，我是李四"
  response1 = conversation_history.invoke(message1, config=config)
  print(f"用户: {message1}")
  print(f"AI: {response1}")
  
  # 第二轮对话
  print("\n--- 第二轮对话 ---")
  message2 = "我想了解一下人工智能"
  response2 = conversation_history.invoke(message2, config=config)
  print(f"用户: {message2}")
  print(f"AI: {response2}")
  
  # 第三轮对话
  print("\n--- 第三轮对话 ---")
  message3 = "能推荐一些学习资源吗？"
  response3 = conversation_history.invoke(message3, config=config)
  print(f"用户: {message3}")
  print(f"AI: {response3}")
  
  print("\n--- 查看对话历史线程状态 ---")
  current_state = conversation_history.get_state(config)
  print(f"当前状态值: {current_state.values}")
  
  print("\n--- 查看对话历史记录 ---")
  history = list(conversation_history.get_state_history(config))
  print(f"历史记录条数: {len(history)}")
  
  for i, state in enumerate(history):
    print(f"\n历史状态 {i+1}:")
    print(f" 值: {state.values}")
    print(f" 创建时间: {state.created_at}")
    print(f" 步骤: {state.metadata.get('step', 'N/A')}")

# 复杂状态管理示例
@entrypoint(checkpointer=InMemorySaver())
def complex_state_manager(updates: Dict[str, Any], *, previous: Optional[Dict[str, Any]] = None) -> entrypoint.final[Dict[str, Any], Dict[str, Any]]:
  """
  复杂状态管理示例
  
  参数:
    updates: 状态更新
    previous: 当前状态
    
  返回:
    entrypoint.final对象，包含返回值和保存值
  """
  print(f"收到状态更新: {updates}")
  
  # 初始化状态
  if previous is None:
    previous = {
      "user_info": {},
      "preferences": {},
      "interactions": 0,
      "last_updated": None
    }
  
  print(f"当前状态: {previous}")
  
  # 更新用户信息
  if "user_info" in updates:
    previous["user_info"].update(updates["user_info"])
  
  # 更新偏好设置
  if "preferences" in updates:
    previous["preferences"].update(updates["preferences"])
  
  # 增加交互次数
  previous["interactions"] += 1
  
  # 更新时间戳
  previous["last_updated"] = time.time()
  
  # 生成状态报告
  report = {
    "message": "状态更新成功",
    "interactions": previous["interactions"],
    "user_name": previous["user_info"].get("name", "未知用户")
  }
  
  print(f"更新后状态: {previous}")
  print(f"生成报告: {report}")
  
  # 返回报告给调用者，保存完整状态
  return entrypoint.final(value=report, save=previous)

def complex_state_demo():
  """复杂状态管理演示"""
  print("\n\n=== 复杂状态管理演示 ===")
  
  config = {"configurable": {"thread_id": "complex-state-thread"}}
  
  # 第一次状态更新
  print("\n--- 第一次状态更新 ---")
  update1 = {
    "user_info": {"name": "王五", "email": "wangwu@example.com"},
    "preferences": {"theme": "dark", "language": "zh-CN"}
  }
  result1 = complex_state_manager.invoke(update1, config=config)
  print(f"更新内容: {update1}")
  print(f"返回结果: {result1}")
  
  # 第二次状态更新
  print("\n--- 第二次状态更新 ---")
  update2 = {
    "preferences": {"notifications": True},
    "user_info": {"age": 30}
  }
  result2 = complex_state_manager.invoke(update2, config=config)
  print(f"更新内容: {update2}")
  print(f"返回结果: {result2}")
  
  # 第三次状态更新
  print("\n--- 第三次状态更新 ---")
  update3 = {
    "user_info": {"location": "上海"},
    "preferences": {"theme": "light"}
  }
  result3 = complex_state_manager.invoke(update3, config=config)
  print(f"更新内容: {update3}")
  print(f"返回结果: {result3}")
  
  print("\n--- 查看复杂状态线程状态 ---")
  current_state = complex_state_manager.get_state(config)
  print(f"当前状态值: {current_state.values}")
  
  print("\n--- 查看复杂状态历史记录 ---")
  history = list(complex_state_manager.get_state_history(config))
  print(f"历史记录条数: {len(history)}")
  
  for i, state in enumerate(history):
    print(f"\n历史状态 {i+1}:")
    print(f" 值: {state.values}")
    print(f" 创建时间: {state.created_at}")
    print(f" 步骤: {state.metadata.get('step', 'N/A')}")

if __name__ == "__main__":
  # 执行所有演示
  accumulate_demo()
  state_management_demo()
  conversation_demo()
  complex_state_demo()