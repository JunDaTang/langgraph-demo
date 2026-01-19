"""
基于LangGraph 1.0函数式API的非确定性控制流简单示例

这个示例说明如何正确处理非确定性操作。
"""

import time
import random
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

# 模拟耗时任务
@task
def slow_task(task_id: int) -> str:
  """模拟耗时任务"""
  return f"任务 {task_id} 完成"

# 正确的方式：将非确定性操作封装在task中
@task
def get_time() -> float: 
  """获取当前时间的任务"""
  return time.time()

# 不正确的非确定性处理方式
@entrypoint(checkpointer=InMemorySaver())
def my_incorrect_workflow(inputs: dict) -> dict:
  """不正确处理时间的工作流"""
  t0 = inputs["t0"]
  # 错误：直接在工作流中获取时间
  t1 = time.time() # 恢复时会重新获取时间，可能导致不同结果

  delta_t = t1 - t0

  if delta_t > 1:
    result = slow_task(1).result()
    value = interrupt("question")
  else:
    result = slow_task(2).result()
    value = interrupt("question")

  return {
    "result": result,
    "value": value
  }

# 正确的非确定性处理方式
@entrypoint(checkpointer=InMemorySaver())
def my_correct_workflow(inputs: dict) -> dict:
  """正确处理时间的工作流"""
  t0 = inputs["t0"]
  # 正确：将获取时间的操作封装在任务中
  t1 = get_time().result() # 恢复时会返回相同的时间值

  delta_t = t1 - t0

  if delta_t > 1:
    result = slow_task(1).result()
    value = interrupt("question")
  else:
    result = slow_task(2).result()
    value = interrupt("question")

  return {
    "result": result,
    "value": value
  }

def demo():
  """演示正确和不正确的非确定性处理方式"""
  print("=== 非确定性控制流处理演示 ===\n")
  
  # 正确方式演示
  print("1. 正确的非确定性处理方式:")
  config = {"configurable": {"thread_id": "correct-demo"}}
  inputs = {"t0": time.time()}
  
  for event in my_correct_workflow.stream(inputs, config):
    print(f"  事件: {event}")
  
  # 恢复执行
  for event in my_correct_workflow.stream(Command(resume="answer"), config):
    print(f"  恢复事件: {event}")
  
  print("\n2. 不正确的非确定性处理方式:")
  config2 = {"configurable": {"thread_id": "incorrect-demo"}}
  inputs2 = {"t0": time.time()}
  
  for event in my_incorrect_workflow.stream(inputs2, config2):
    print(f"  事件: {event}")
  
  # 恢复执行（非确定性操作会重新执行）
  for event in my_incorrect_workflow.stream(Command(resume="answer"), config2):
    print(f"  恢复事件: {event}")
    print("  注意：时间会被重新获取，可能导致不同分支执行！")

if __name__ == "__main__":
  demo()