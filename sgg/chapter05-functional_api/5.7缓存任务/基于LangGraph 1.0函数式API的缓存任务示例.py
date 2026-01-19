"""
基于LangGraph 1.0函数式API的缓存任务示例

本示例演示了如何在LangGraph函数式API中配置和使用缓存策略，
展示了任务结果的缓存机制以提高性能。

主要特性：
1. 使用CachePolicy配置任务缓存策略
2. 使用InMemoryCache作为缓存存储
3. 使用@task装饰器定义带缓存策略的任务函数
4. 使用@entrypoint装饰器定义带缓存的工作流入口点
"""

import time
from langgraph.cache.memory import InMemoryCache
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy

# 定义带缓存策略的任务函数
@task(cache_policy=CachePolicy(ttl=120)) # 设置120秒的缓存时间
def slow_add(x: int) -> int:
  """
  模拟耗时的加法运算任务
  
  Args:
    x: 输入数字
    
  Returns:
    输入数字的两倍
  """
  print(f"执行耗时运算: {x} * 2")
  time.sleep(1) # 模拟耗时操作
  result = x * 2
  print(f"运算完成，结果: {result}")
  return result

# 定义带缓存的工作流入口点
@entrypoint(cache=InMemoryCache())
def main(inputs: dict) -> dict[str, int]:
  """
  主工作流，演示缓存机制
  
  Args:
    inputs: 输入数据字典
    
  Returns:
    包含两次运算结果的字典
  """
  print("启动主工作流")
  
  # 第一次调用slow_add任务
  print("第一次调用slow_add任务")
  result1 = slow_add(inputs["x"]).result()
  
  # 第二次调用slow_add任务（应该从缓存中获取结果）
  print("第二次调用slow_add任务")
  result2 = slow_add(inputs["x"]).result()
  
  return {"result1": result1, "result2": result2}

# 更复杂的缓存示例
@task(cache_policy=CachePolicy(ttl=60)) # 设置60秒的缓存时间
def complex_calculation(data: dict) -> dict:
  """
  复杂计算任务
  
  Args:
    data: 输入数据字典
    
  Returns:
    计算结果字典
  """
  print(f"执行复杂计算: {data}")
  time.sleep(2) # 模拟复杂计算
  
  # 执行一些计算
  result = {
    "sum": sum(data.values()),
    "count": len(data),
    "average": sum(data.values()) / len(data) if data else 0
  }
  
  print(f"复杂计算完成，结果: {result}")
  return result

@entrypoint(cache=InMemoryCache())
def complex_workflow(inputs: dict) -> dict:
  """
  复杂工作流，演示更复杂的缓存场景
  
  Args:
    inputs: 输入数据字典
    
  Returns:
    处理结果字典
  """
  print("启动复杂工作流")
  
  # 第一次调用复杂计算任务
  print("第一次调用复杂计算任务")
  result1 = complex_calculation(inputs["data"]).result()
  
  # 第二次调用复杂计算任务（应该从缓存中获取结果）
  print("第二次调用复杂计算任务")
  result2 = complex_calculation(inputs["data"]).result()
  
  # 修改输入数据后再次调用
  modified_data = {k: v + 1 for k, v in inputs["data"].items()}
  print(f"使用修改后的数据调用复杂计算任务: {modified_data}")
  result3 = complex_calculation(modified_data).result()
  
  return {
    "first_call": result1,
    "second_call": result2,
    "third_call": result3
  }

def main_demo():
  """主演示函数"""
  print("=== LangGraph 1.0 函数式API缓存任务演示 ===")
  
  print("\n--- 执行主工作流 ---")
  # 流式执行主工作流
  for chunk in main.stream({"x": 5}, stream_mode="updates"):
    print(f"流式输出: {chunk}")

def complex_demo():
  """复杂缓存演示"""
  print("\n\n=== 复杂缓存演示 ===")
  
  print("\n--- 执行复杂工作流 ---")
  # 流式执行复杂工作流
  for chunk in complex_workflow.stream(
    {"data": {"a": 10, "b": 20, "c": 30}}, 
    stream_mode="updates"
  ):
    print(f"流式输出: {chunk}")


# 带有过期时间的缓存示例
@task(cache_policy=CachePolicy(ttl=3)) # 设置3秒的缓存时间
def short_ttl_task(x: int) -> int:
  """
  短缓存时间任务
  
  Args:
    x: 输入数字
    
  Returns:
    输入数字的平方
  """
  print(f"执行短缓存时间任务: {x}^2")
  time.sleep(0.5) # 模拟较短的耗时操作
  result = x ** 2
  print(f"短缓存时间任务完成，结果: {result}")
  return result

@entrypoint(cache=InMemoryCache())
def ttl_workflow(inputs: dict) -> dict:
  """
  演示缓存过期的工作流
  
  Args:
    inputs: 输入数据字典
    
  Returns:
    处理结果字典
  """
  print("启动缓存过期演示工作流")
  
  # 第一次调用
  print("第一次调用短缓存时间任务")
  result1 = short_ttl_task(inputs["x"]).result()
  
  # 等待缓存过期
  print("等待缓存过期（4秒）...")
  time.sleep(4)
  
  # 第二次调用（缓存已过期，需要重新计算）
  print("第二次调用短缓存时间任务（缓存已过期）")
  result2 = short_ttl_task(inputs["x"]).result()
  
  return {
    "first_result": result1,
    "second_result": result2
  }

def ttl_demo():
  """缓存过期演示"""
  print("\n\n=== 缓存过期演示 ===")
  
  print("\n--- 执行缓存过期工作流 ---")
  for chunk in ttl_workflow.stream({"x": 3}, stream_mode="updates"):
    print(f"流式输出: {chunk}")

if __name__ == "__main__":
  # 执行所有演示
  main_demo()
  complex_demo()
  ttl_demo()