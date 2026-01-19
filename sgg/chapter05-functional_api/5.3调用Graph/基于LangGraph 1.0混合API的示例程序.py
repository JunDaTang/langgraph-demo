"""
基于LangGraph 1.0混合API的示例程序

本示例演示了如何在同一应用程序中同时使用函数式API和图API，
展示了两种API共享相同底层运行时的特点。

主要特性：
1. 使用图API定义状态图和节点
2. 使用函数式API定义工作流入口点
3. 在函数式API中调用图API构建的状态图
4. 使用InMemorySaver进行状态持久化
"""

import uuid
from typing import TypedDict
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

# 定义共享状态类型
class State(TypedDict):
  """状态定义，包含一个整数字段foo"""
  foo: int

# 定义简单的转换节点
def double(state: State) -> State:
  """
  将状态中的foo值翻倍
  
  Args:
    state: 当前状态
    
  Returns:
    更新后的状态
  """
  print(f"执行double节点: {state['foo']} * 2 = {state['foo'] * 2}")
  return {"foo": state["foo"] * 2}

def add_five(state: State) -> State:
  """
  将状态中的foo值加5
  
  Args:
    state: 当前状态
    
  Returns:
    更新后的状态
  """
  print(f"执行add_five节点: {state['foo']} + 5 = {state['foo'] + 5}")
  return {"foo": state["foo"] + 5}

# 使用图API构建状态图
def build_graph():
  """
  构建一个简单的状态图
  
  Returns:
    编译后的状态图
  """
  builder = StateGraph(State)
  builder.add_node("double", double)
  builder.add_node("add_five", add_five)
  builder.add_edge(START, "double")
  builder.add_edge("double", "add_five")
  builder.add_edge("add_five", END)
  return builder.compile()

# 创建图实例
graph = build_graph()

# 定义函数式API任务 - 处理单个数字
@task
def process_number(x: int) -> int:
  """
  使用图API处理单个数字
  
  Args:
    x: 输入数字
    
  Returns:
    处理后的结果
  """
  result = graph.invoke({"foo": x})
  return result["foo"]

# 定义函数式API工作流入口点
@entrypoint(checkpointer=InMemorySaver())
def workflow(numbers: list[int]) -> dict:
  """
  处理数字列表的工作流
  
  Args:
    numbers: 数字列表
    
  Returns:
    处理结果字典
  """
  print(f"开始处理数字列表: {numbers}")
  
  # 并行处理所有数字
  futures = [process_number(num) for num in numbers]
  results = [f.result() for f in futures]
  
  return {
    "input_numbers": numbers,
    "output_numbers": results,
    "total": sum(results)
  }

def main():
  """主函数，演示混合API的使用"""
  print("=== LangGraph 1.0 混合API使用示例 ===")
  
  # 定义输入数据
  numbers = [1, 2, 3, 4, 5]
  print(f"输入数字列表: {numbers}")
  
  # 生成唯一线程ID用于状态保存
  thread_id = str(uuid.uuid4())
  config = {"configurable": {"thread_id": thread_id}}
  print(f"工作流线程ID: {thread_id}")
  
  # 执行工作流
  print("\n--- 开始执行混合API工作流 ---")
  result = workflow.invoke(numbers, config=config)
  
  # 输出结果
  print("\n--- 处理结果 ---")
  print(f"输入数字: {result['input_numbers']}")
  print(f"输出数字: {result['output_numbers']}")
  print(f"总和: {result['total']}")
  
  # 单独演示图API的使用
  print("\n--- 单独演示图API ---")
  graph_result = graph.invoke({"foo": 10})
  print(f"图API处理结果: {graph_result}") # 应该输出: {'foo': 25} (10*2+5)

# 为了更好地理解两种API的关系，我们再创建一个更直接的混合示例
@entrypoint(checkpointer=InMemorySaver())
def simple_workflow(x: int) -> dict:
  """
  简单的工作流示例，直接在函数式API中调用图API
  
  Args:
    x: 输入数字
    
  Returns:
    处理结果
  """
  # 直接调用图API构建的状态图
  result = graph.invoke({"foo": x})
  return {"bar": result["foo"]}

def simple_demo():
  """简单混合API演示"""
  print("\n=== 简单混合API演示 ===")
  
  # 生成唯一线程ID用于状态保存
  thread_id = str(uuid.uuid4())
  config = {"configurable": {"thread_id": thread_id}}
  print(f"工作流线程ID: {thread_id}")
  
  # 执行简单工作流
  result = simple_workflow.invoke(5, config=config)
  print(f"简单工作流结果: {result}") # 应该输出: {'bar': 25}

# 参考资料中的直接示例
@entrypoint(checkpointer=InMemorySaver())
def reference_workflow(x: int) -> dict:
  """
  参考资料中的示例工作流
  
  Args:
    x: 输入数字
    
  Returns:
    处理结果
  """
  result = graph.invoke({"foo": x})
  return {"bar": result["foo"]}

def reference_demo():
  """参考资料示例演示"""
  print("\n=== 参考资料示例演示 ===")
  
  # 生成唯一线程ID用于状态保存
  thread_id = str(uuid.uuid4())
  config = {"configurable": {"thread_id": thread_id}}
  print(f"工作流线程ID: {thread_id}")
  
  # 执行参考资料中的工作流
  result = reference_workflow.invoke(5, config=config)
  print(f"参考资料工作流结果: {result}") # 应该输出: {'bar': 25}

if __name__ == "__main__":
  main()
  simple_demo()
  reference_demo()