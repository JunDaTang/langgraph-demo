"""
基于LangGraph 1.0函数式API的流式处理示例

本示例演示了如何使用LangGraph函数式API的流式处理机制，
展示了与Graph API相同的流机制。

主要特性：
1. 使用@entrypoint装饰器定义支持流式处理的工作流
2. 使用get_stream_writer获取流写入器
3. 通过流写入器发送自定义流数据
4. 使用stream_mode参数控制流式输出模式
"""

from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.types import StreamWriter

# 初始化检查点保存器
checkpointer = InMemorySaver()

# 定义一个任务函数 - 模拟耗时操作
@task
def process_data(data: str) -> str:
  """
  处理数据的任务函数

  Args:
    data: 输入数据

  Returns:
    处理后的数据
  """
  # 模拟数据处理
  processed = f"[已处理] {data}"
  return processed

# 定义支持流式处理的主工作流
@entrypoint(checkpointer=checkpointer)
def main(inputs: dict,writer:StreamWriter) -> dict:
  """
  支持流式处理的主工作流

  Args:
    inputs: 输入数据字典

  Returns:
    处理结果字典
  """

  # 发送开始处理消息
  writer("开始处理数据")

  # 获取输入数据
  x = inputs["x"]

  # 发送处理进度消息
  writer(f"正在处理数据: {x}")

  # 执行处理操作
  result = x * 2

  # 发送中间结果消息
  writer(f"中间结果: {result}")

  # 使用任务函数处理数据
  processed_data = process_data(f"数据_{x}").result()

  # 发送最终结果消息
  writer(f"处理完成，最终结果: {result}")

  return {
    "input": x,
    "output": result,
    "processed_data": processed_data
  }

# 更复杂的流式处理示例
@entrypoint(checkpointer=checkpointer)
def complex_workflow(inputs: dict) -> dict:
  """
  复杂的流式处理工作流

  Args:
    inputs: 输入数据字典

  Returns:
    处理结果字典
  """
  # 获取流写入器
  writer = get_stream_writer()

  # 发送开始消息
  writer("启动复杂工作流")

  # 获取输入数据
  numbers = inputs["numbers"]
  writer(f"待处理数字列表: {numbers}")

  # 逐步处理每个数字
  results = []
  for i, num in enumerate(numbers):
    writer(f"正在处理第 {i + 1} 个数字: {num}")
    processed = num ** 2 # 平方运算
    results.append(processed)
    writer(f"第 {i + 1} 个数字处理完成，结果: {processed}")

  # 计算总和
  total = sum(results)
  writer(f"所有数字处理完成，总和: {total}")

  return {
    "input_numbers": numbers,
    "squared_numbers": results,
    "total": total
  }

def main_demo():
  """主工作流演示"""
  print("=== LangGraph 1.0 函数式API流式处理演示 ===")

  # 配置
  config = {"configurable": {"thread_id": "main-demo-123"}}

  print("\n--- 执行主工作流并流式输出 ---")
  # 流式执行工作流
  for mode, chunk in main.stream(
      {"x": 5},
      stream_mode=["custom", "values"],
      config=config
  ):
    print(f"[{mode}]: {chunk}")
# 并行任务流式处理示例
@task
def square_task(x: int) -> int:
  """
  计算平方的任务

  Args:
    x: 输入数字

  Returns:
    平方结果
  """
  return x ** 2

@task
def cube_task(x: int) -> int:
  """
  计算立方的任务

  Args:
    x: 输入数字

  Returns:
    立方结果
  """
  return x ** 3

@entrypoint(checkpointer=checkpointer)
def parallel_workflow(inputs: dict) -> dict:
  """
  并行任务流式处理工作流

  Args:
    inputs: 输入数据字典

  Returns:
    处理结果字典
  """
  # 获取流写入器
  writer = get_stream_writer()
  writer("启动并行任务工作流")

  # 获取输入数据
  num = inputs["number"]
  writer(f"待处理数字: {num}")

  # 并行启动任务
  writer("开始并行执行平方和立方计算任务")
  square_future = square_task(num)
  cube_future = cube_task(num)

  # 等待任务完成
  square_result = square_future.result()
  cube_result = cube_future.result()

  writer(f"平方计算结果: {square_result}")
  writer(f"立方计算结果: {cube_result}")

  return {
    "input": num,
    "square": square_result,
    "cube": cube_result
  }

def parallel_demo():
  """并行任务工作流演示"""
  print("\n\n=== 并行任务流式处理演示 ===")

  config = {"configurable": {"thread_id": "parallel-demo-789"}}

  print("\n--- 执行并行任务工作流并流式输出 ---")
  for mode, chunk in parallel_workflow.stream(
      {"number": 3},
      stream_mode=["custom", "updates"],
      config=config
  ):
    print(f"[{mode}]: {chunk}")

if __name__ == "__main__":
  # 执行所有演示
  main_demo() # 基本的流式处理功能
  parallel_demo() # 并行任务流式处理功能