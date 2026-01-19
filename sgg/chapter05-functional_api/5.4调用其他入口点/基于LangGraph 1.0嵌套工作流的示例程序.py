"""
基于LangGraph 1.0嵌套工作流的示例程序

本示例演示了如何在一个入口点或任务中调用其他入口点，
展示了LangGraph函数式API的嵌套工作流功能。

主要特性：
1. 使用@entrypoint装饰器定义可复用的子工作流
2. 在主工作流中调用子工作流
3. 使用InMemorySaver进行状态持久化
"""

import uuid
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

# 初始化检查点保存器
checkpointer = InMemorySaver()

# 可复用的子工作流 - 执行乘法运算
@entrypoint()
def multiply(inputs: dict) -> int:
  """
  执行乘法运算的子工作流
  
  Args:
    inputs: 包含操作数的字典，必须包含"a"和"b"键
    
  Returns:
    两个数的乘积
  """
  a = inputs["a"]
  b = inputs["b"]
  result = a * b
  print(f"执行乘法运算: {a} × {b} = {result}")
  return result

# 可复用的子工作流 - 执行加法运算
@entrypoint()
def add(inputs: dict) -> int:
  """
  执行加法运算的子工作流
  
  Args:
    inputs: 包含操作数的字典，必须包含"a"和"b"键
    
  Returns:
    两个数的和
  """
  a = inputs["a"]
  b = inputs["b"]
  result = a + b
  print(f"执行加法运算: {a} + {b} = {result}")
  return result

# 可复用的子工作流 - 执行幂运算
@entrypoint()
def power(inputs: dict) -> int:
  """
  执行幂运算的子工作流
  
  Args:
    inputs: 包含底数和指数的字典，必须包含"base"和"exp"键
    
  Returns:
    底数的指数次幂
  """
  base = inputs["base"]
  exp = inputs["exp"]
  result = base ** exp
  print(f"执行幂运算: {base} ^ {exp} = {result}")
  return result

# 主工作流 - 调用多个子工作流
@entrypoint(checkpointer=checkpointer)
def main(inputs: dict) -> dict:
  """
  主工作流，调用多个子工作流执行复杂计算
  
  Args:
    inputs: 包含计算参数的字典
    
  Returns:
    计算结果字典
  """
  x = inputs["x"]
  y = inputs["y"]
  z = inputs["z"]
  
  print(f"开始执行主工作流，输入参数: x={x}, y={y}, z={z}")
  
  # 调用乘法子工作流
  product = multiply.invoke({"a": x, "b": y})
  
  # 调用加法子工作流
  sum_result = add.invoke({"a": product, "b": z})
  
  # 调用幂运算子工作流
  power_result = power.invoke({"base": sum_result, "exp": 2})
  
  return {
    "product": product,      # x * y
    "sum": sum_result,      # (x * y) + z
    "power": power_result,    # ((x * y) + z) ^ 2
    "final_result": power_result # 最终结果
  }

# 更复杂的嵌套示例 - 递归调用工作流
@entrypoint(checkpointer=checkpointer)
def factorial(inputs: dict) -> int:
  """
  计算阶乘的递归工作流
  
  Args:
    inputs: 包含数字n的字典
    
  Returns:
    n的阶乘
  """
  n = inputs["n"]
  
  # 基础情况
  if n <= 1:
    print(f"阶乘基础情况: {n}! = 1")
    return 1
  
  # 递归情况
  print(f"计算阶乘: {n}! = {n} × {n-1}!")
  sub_result = factorial.invoke({"n": n - 1})
  result = n * sub_result
  print(f"阶乘结果: {n}! = {result}")
  return result

# 组合工作流 - 调用阶乘工作流
@entrypoint(checkpointer=checkpointer)
def combination_workflow(inputs: dict) -> dict:
  """
  组合工作流，计算多个数的阶乘
  
  Args:
    inputs: 包含数字列表的字典
    
  Returns:
    阶乘计算结果字典
  """
  numbers = inputs["numbers"]
  print(f"开始计算数字列表 {numbers} 的阶乘")
  
  # 并行计算每个数的阶乘
  factorial_results = []
  for num in numbers:
    fact = factorial.invoke({"n": num})
    factorial_results.append(fact)
  
  return {
    "input_numbers": numbers,
    "factorials": factorial_results,
    "results_dict": dict(zip(numbers, factorial_results))
  }

def main_demo():
  """主工作流演示"""
  print("=== LangGraph 1.0 嵌套工作流演示 ===")
  
  # 生成唯一线程ID用于状态保存
  thread_id = str(uuid.uuid4())
  config = {"configurable": {"thread_id": thread_id}}
  print(f"主工作流线程ID: {thread_id}")
  
  # 执行主工作流
  print("\n--- 执行主工作流 ---")
  inputs = {"x": 6, "y": 7, "z": 5}
  result = main.invoke(inputs, config=config)
  
  # 输出结果
  print("\n--- 计算结果 ---")
  print(f"输入: x={inputs['x']}, y={inputs['y']}, z={inputs['z']}")
  print(f"乘法结果 (x × y): {result['product']}")
  print(f"加法结果 ((x × y) + z): {result['sum']}")
  print(f"幂运算结果 (((x × y) + z) ^ 2): {result['power']}")
  print(f"最终结果: {result['final_result']}")

def factorial_demo():
  """阶乘工作流演示"""
  print("\n\n=== 阶乘工作流演示 ===")

  # 生成唯一线程ID用于状态保存
  thread_id = str(uuid.uuid4())
  config = {"configurable": {"thread_id": thread_id}}
  print(f"阶乘工作流线程ID: {thread_id}")

  # 执行组合工作流
  print("\n--- 执行组合工作流 ---")
  inputs = {"numbers": [3, 4, 5]}
  result = combination_workflow.invoke(inputs, config=config)

  # 输出结果
  print("\n--- 阶乘计算结果 ---")
  print(f"输入数字: {result['input_numbers']}")
  print(f"阶乘结果: {result['factorials']}")
  print("详细结果:")
  for num, fact in result['results_dict'].items():
    print(f" {num}! = {fact}")


if __name__ == "__main__":
  main_demo() #在主工作流中调用多个子工作流执行复杂计算
  factorial_demo() #递归调用工作流的功能