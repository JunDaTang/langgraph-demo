"""
LangGraph 节点重试策略演示
"""

import random
from typing import Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

# 定义状态
class State(TypedDict):
  result: str

# 模拟不稳定的API调用，使用全局变量跟踪尝试次数
attempt_counter = 0

def unstable_api_call(state: State) -> Dict[str, Any]:
  """
  模拟一个不稳定的API调用，有一定概率失败
  """
  global attempt_counter
  attempt_counter += 1
  print(f"尝试调用API，这是第 {attempt_counter} 次尝试")
  
  # 模拟前几次尝试失败，最后一次成功
  if attempt_counter < 3:
    raise Exception(f"模拟API调用失败 (尝试 {attempt_counter})")
  else:
    # 第三次尝试成功
    return {
      "result": f"API调用成功，经过 {attempt_counter} 次尝试"
    }

# 自定义重试策略
def custom_retry_on(exception: Exception) -> bool:
  """
  自定义重试条件：只对特定错误进行重试
  """
  # 只对包含"模拟API调用失败"的异常进行重试
  if "模拟API调用失败" in str(exception):
    print(f"捕获到可重试异常: {exception}")
    return True
  
  # 对其他异常不重试
  print(f"捕获到不可重试异常: {exception}")
  return False

# 模拟抛出 ValueError 的节点
def value_error_call(state: State) -> Dict[str, Any]:
  """
  模拟抛出 ValueError 的节点（不会被默认重试策略重试）
  """
  print("调用会抛出 ValueError 的节点")
  raise ValueError("模拟 ValueError 异常")

def run_demo():
  print("=== LangGraph 节点重试策略演示 ===\n")
  
  # 重置全局计数器
  global attempt_counter
  attempt_counter = 0
  
  # 演示1: 使用默认重试策略
  print("1. 使用默认重试策略:")
  print("  默认策略会对除特定异常外的所有异常进行重试")
  print("  不会重试的异常包括: ValueError, TypeError, ArithmeticError, ImportError,")
  print("           LookupError, NameError, SyntaxError, RuntimeError,")
  print("           ReferenceError, StopIteration, StopAsyncIteration, OSError\n")
  
  builder1 = StateGraph(State)
  
  # 添加节点，使用默认重试策略
  builder1.add_node(
    "unstable_call", 
    unstable_api_call, 
    retry_policy=RetryPolicy(max_attempts=5) # 允许最多5次尝试
  )
  
  builder1.add_edge(START, "unstable_call")
  builder1.add_edge("unstable_call", END)
  
  graph1 = builder1.compile()
  
  print("测试默认重试策略:")
  try:
    result = graph1.invoke({"result": ""})
    print(f"最终结果: {result}\n")
  except Exception as e:
    print(f"最终失败: {type(e).__name__}: {e}\n")
  
  # 演示2: 使用自定义重试策略
  print("2. 使用自定义重试策略:")
  print("  自定义策略只对特定错误进行重试\n")
  
  # 重置全局计数器
  attempt_counter = 0
  
  builder2 = StateGraph(State)
  
  # 添加节点，使用自定义重试策略
  builder2.add_node(
    "custom_retry_call", 
    unstable_api_call, 
    retry_policy=RetryPolicy(max_attempts=5, retry_on=custom_retry_on)
  )
  
  builder2.add_edge(START, "custom_retry_call")
  builder2.add_edge("custom_retry_call", END)
  
  graph2 = builder2.compile()
  
  print("测试自定义重试策略:")
  try:
    result = graph2.invoke({"result": ""})
    print(f"最终结果: {result}\n")
  except Exception as e:
    print(f"最终失败: {type(e).__name__}: {e}\n")
  
  # 演示3: 不会重试的异常类型
  print("3. 测试不会重试的异常类型:")
  
  builder3 = StateGraph(State)
  
  # 添加节点，使用默认重试策略
  builder3.add_node(
    "value_error_call", 
    value_error_call, 
    retry_policy=RetryPolicy(max_attempts=3)
  )
  
  builder3.add_edge(START, "value_error_call")
  builder3.add_edge("value_error_call", END)
  
  graph3 = builder3.compile()
  
  print("测试 ValueError（默认策略不会重试）:")
  try:
    result = graph3.invoke({"result": ""})
    print(f"最终结果: {result}\n")
  except Exception as e:
    print(f"最终失败: {type(e).__name__}: {e}\n")

if __name__ == "__main__":
  run_demo()