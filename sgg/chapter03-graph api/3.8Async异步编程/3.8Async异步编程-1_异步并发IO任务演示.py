"""
LangGraph 异步并发IO任务演示

本示例展示了如何使用LangGraph 1.0版本和异步编程范式，
在处理IO密集型任务（如并发API请求）时显著提升性能。
"""

import asyncio
import time
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class IOState(TypedDict):
  """
  状态定义，用于跟踪并发IO任务的结果
  """
  results: Annotated[list, lambda x, y: x + y] # 合并列表结果
  task_count: int # 总任务数

async def simulated_api_call(task_id: int, delay: float = 1.0) -> str:
  """
  模拟API调用的异步函数
  
  Args:
    task_id: 任务ID
    delay: 延迟时间（秒），模拟网络请求耗时
  
  Returns:
    str: API响应结果
  """
  print(f"开始执行API调用 {task_id}...")
  await asyncio.sleep(delay) # 模拟网络IO延迟
  result = f"API调用 {task_id} 的结果"
  print(f"完成API调用 {task_id}")
  return result

async def io_task_1(state: IOState) -> dict:
  """
  IO密集型任务1 - 模拟API调用
  """
  result = await simulated_api_call(1, 1.0)
  return {"results": [result]}

async def io_task_2(state: IOState) -> dict:
  """
  IO密集型任务2 - 模拟API调用
  """
  result = await simulated_api_call(2, 1.5)
  return {"results": [result]}

async def io_task_3(state: IOState) -> dict:
  """
  IO密集型任务3 - 模拟API调用
  """
  result = await simulated_api_call(3, 0.8)
  return {"results": [result]}

async def io_task_4(state: IOState) -> dict:
  """
  IO密集型任务4 - 模拟API调用
  """
  result = await simulated_api_call(4, 1.2)
  return {"results": [result]}

def summary_node(state: IOState) -> dict:
  """
  汇总节点 - 收集所有并发任务的结果
  """
  print(f"\n=== 所有任务已完成 ===")
  print(f"总共执行了 {state['task_count']} 个任务")
  print(f"收集到 {len(state['results'])} 个结果:")
  for i, result in enumerate(state['results'], 1):
    print(f" {i}. {result}")
  
  return {"results": state["results"] + ["所有任务已完成"]}

# 同步版本用于对比性能
def sync_version():
  """
  同步版本用于性能对比
  """
  print("=== 同步执行版本 ===")
  start_time = time.time()
  
  results = []
  delays = [1.0, 1.5, 0.8, 1.2]
  
  for i, delay in enumerate(delays, 1):
    print(f"开始执行同步任务 {i}...")
    time.sleep(delay) # 模拟阻塞IO
    result = f"同步任务 {i} 的结果"
    results.append(result)
    print(f"完成同步任务 {i}")
  
  elapsed = time.time() - start_time
  print(f"\n同步执行总耗时: {elapsed:.2f} 秒")
  return results

async def async_version():
  """
  异步版本用于性能对比
  """
  print("\n=== 异步执行版本 ===")
  start_time = time.time()
  
  # 创建所有任务
  tasks = [
    simulated_api_call(1, 1.0),
    simulated_api_call(2, 1.5),
    simulated_api_call(3, 0.8),
    simulated_api_call(4, 1.2)
  ]
  
  # 并发执行所有任务
  results = await asyncio.gather(*tasks)
  
  elapsed = time.time() - start_time
  print(f"\n异步执行总耗时: {elapsed:.2f} 秒")
  return results

def build_async_graph():
  """
  构建LangGraph异步工作流
  """
  builder = StateGraph(IOState)
  
  # 添加节点
  builder.add_node("io_task_1", io_task_1)
  builder.add_node("io_task_2", io_task_2)
  builder.add_node("io_task_3", io_task_3)
  builder.add_node("io_task_4", io_task_4)
  builder.add_node("summary", summary_node)
  
  # 添加边 - 所有IO任务从START并行开始
  builder.add_edge(START, "io_task_1")
  builder.add_edge(START, "io_task_2")
  builder.add_edge(START, "io_task_3")
  builder.add_edge(START, "io_task_4")
  
  # 所有IO任务完成后汇聚到summary节点
  builder.add_edge("io_task_1", "summary")
  builder.add_edge("io_task_2", "summary")
  builder.add_edge("io_task_3", "summary")
  builder.add_edge("io_task_4", "summary")
  builder.add_edge("summary", END)
  
  # 编译图
  return builder.compile()

async def main():
  """
  主函数 - 演示异步并发IO任务的优势
  """
  print("LangGraph 异步并发IO任务演示")
  print("=" * 50)
  
  # 1. 先展示传统的同步执行方式
  sync_results = sync_version()
  
  # 2. 展示纯异步执行方式
  async_results = await async_version()
  
  # 3. 展示使用LangGraph的异步工作流
  print("\n=== LangGraph异步工作流版本 ===")
  start_time = time.time()
  
  # 构建异步图
  graph = build_async_graph()
  
  # 执行图
  result = await graph.ainvoke({
    "results": [],
    "task_count": 4
  })
  
  elapsed = time.time() - start_time
  print(f"\nLangGraph异步工作流总耗时: {elapsed:.2f} 秒")
  
  # 4. 性能对比总结
  print("\n=== 性能对比总结 ===")
  print("通过异步并发执行IO密集型任务，可以显著提升性能:")
  print("- 同步执行: 任务依次执行，总耗时为各任务耗时之和")
  print("- 异步执行: 任务并发执行，总耗时近似于最耗时的任务")
  print("- LangGraph异步工作流: 提供了结构化的异步任务编排能力")

if __name__ == "__main__":
  # 运行主函数
  asyncio.run(main())



# LangGraph 异步并发IO任务演示
# ==================================================
# === 同步执行版本 ===
# 开始执行同步任务 1...
# 完成同步任务 1
# 开始执行同步任务 2...
# 完成同步任务 2
# 开始执行同步任务 3...
# 完成同步任务 3
# 开始执行同步任务 4...
# 完成同步任务 4

# 同步执行总耗时: 4.51 秒

# === 异步执行版本 ===
# 开始执行API调用 1...
# 开始执行API调用 2...
# 开始执行API调用 3...
# 开始执行API调用 4...
# 完成API调用 3
# 完成API调用 1
# 完成API调用 4
# 完成API调用 2

# 异步执行总耗时: 1.51 秒

# === LangGraph异步工作流版本 ===
# 开始执行API调用 1...
# 开始执行API调用 2...
# 开始执行API调用 3...
# 开始执行API调用 4...
# 完成API调用 3
# 完成API调用 1
# 完成API调用 4
# 完成API调用 2

# === 所有任务已完成 ===
# 总共执行了 4 个任务
# 收集到 4 个结果:
#  1. API调用 1 的结果
#  2. API调用 2 的结果
#  3. API调用 3 的结果
#  4. API调用 4 的结果

# LangGraph异步工作流总耗时: 1.53 秒

# === 性能对比总结 ===
# 通过异步并发执行IO密集型任务，可以显著提升性能:
# - 同步执行: 任务依次执行，总耗时为各任务耗时之和
# - 异步执行: 任务并发执行，总耗时近似于最耗时的任务
# - LangGraph异步工作流: 提供了结构化的异步任务编排能力
