"""
LangGraph Reducer函数演示 - 自定义 mul reducer 实现数值相乘
使用全局变量区分初始化调用和正常调用
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 使用全局变量来跟踪是否是第一次调用（初始化阶段）
_is_initial_call = True

def my_mul_reducer(current_value: float, new_value: float) -> float:
  """
  自定义乘法reducer，使用全局变量区分初始化调用和正常调用
  
  Args:
    current_value: 当前状态值
    new_value: 新值
    
  Returns:
    计算后的结果
  """
  global _is_initial_call
  
  print(f"Reducer被调用: current_value={current_value}, new_value={new_value}, is_initial_call={_is_initial_call}")
  
  # 如果是初始化调用，直接返回new_value，避免默认值0的影响
  if _is_initial_call:
    _is_initial_call = False # 重置标志
    return new_value
  else:
    # 正常的乘法操作，包括乘以0的情况
    return current_value * new_value

class MultiplyState(TypedDict):
  factor: Annotated[float, my_mul_reducer]

def multiplier_by_two(state: MultiplyState) -> dict:
  """将factor乘以2"""
  return {"factor": 2.0}

def multiplier_by_zero(state: MultiplyState) -> dict:
  """将factor乘以0"""
  return {"factor": 0.0}

def run_demo():
  """
  演示增强版乘法reducer的使用
  """
  global _is_initial_call
  
  print("=== operator.mul 增强版解决方案演示 ===\n")
  
  # 演示1: 正常乘法操作
  print("1. 正常乘法操作演示:")
  _is_initial_call = True # 重置初始化标志
  builder = StateGraph(MultiplyState)
  builder.add_node("multiplier_by_two", multiplier_by_two)
  builder.add_edge(START, "multiplier_by_two")
  builder.add_edge("multiplier_by_two", END)
  graph = builder.compile()

  result = graph.invoke({"factor": 5.0})
  print(f"初始状态: {{'factor': 5.0}}")
  print(f"执行结果: {result}")
  print(f"预期结果: 10.0 (5.0 * 2.0)\n")
  
  # 演示2: 乘以0的操作
  print("2. 乘以0的操作演示:")
  _is_initial_call = True # 重置初始化标志
  builder2 = StateGraph(MultiplyState)
  builder2.add_node("multiplier_by_zero", multiplier_by_zero)
  builder2.add_edge(START, "multiplier_by_zero")
  builder2.add_edge("multiplier_by_zero", END)
  graph2 = builder2.compile()

  result2 = graph2.invoke({"factor": 5.0})
  print(f"初始状态: {{'factor': 5.0}}")
  print(f"执行结果: {result2}")
  print(f"预期结果: 0.0 (5.0 * 0.0)\n")
  
  # 演示3: 连续乘法操作
  print("3. 连续乘法操作演示:")
  _is_initial_call = True # 重置初始化标志
  builder3 = StateGraph(MultiplyState)
  builder3.add_node("multiplier_by_two_1", multiplier_by_two)
  builder3.add_node("multiplier_by_zero", multiplier_by_zero)
  builder3.add_node("multiplier_by_two_2", multiplier_by_two)
  
  builder3.add_edge(START, "multiplier_by_two_1")
  builder3.add_edge("multiplier_by_two_1", "multiplier_by_zero")
  builder3.add_edge("multiplier_by_zero", "multiplier_by_two_2")
  builder3.add_edge("multiplier_by_two_2", END)
  
  graph3 = builder3.compile()

  result3 = graph3.invoke({"factor": 3.0})
  print(f"初始状态: {{'factor': 3.0}}")
  print(f"执行结果: {result3}")
  print(f"预期过程: 3.0 -> 6.0 -> 0.0 -> 0.0")
  print(f"预期结果: 0.0\n")

if __name__ == "__main__":
  run_demo()