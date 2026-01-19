"""
基于LangGraph 1.0函数式API的重试策略示例

本示例演示了如何在LangGraph函数式API中配置和使用重试策略，
展示了节点执行失败时的自动重试机制。

主要特性：
1. 使用RetryPolicy配置重试策略
2. 使用@task装饰器定义带重试策略的任务函数
3. 模拟网络故障并演示重试机制
4. 使用InMemorySaver进行状态持久化
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy
import time
import random

# 全局变量仅用于演示目的，模拟网络故障
# 实际代码中不会使用这样的变量
attempts = 0
api_call_attempts = 0
database_attempts = 0

# 配置重试策略以重试ValueError异常
# 默认的RetryPolicy针对特定的网络错误进行了优化
retry_policy = RetryPolicy(retry_on=ValueError)

@task(retry_policy=retry_policy)
def get_info():
  """
  模拟可能失败的信息获取任务
  
  Returns:
    成功时返回"OK"
    
  Raises:
    ValueError: 模拟网络故障
  """
  global attempts
  attempts += 1
  print(f"尝试获取信息，第 {attempts} 次尝试")
  
  # 模拟第一次调用失败
  if attempts < 2:
    print("发生网络故障，抛出ValueError异常")
    raise ValueError('网络连接失败')
  
  print("信息获取成功")
  return "OK"

# 更实际的重试策略示例 - 模拟API调用
api_retry_policy = RetryPolicy(
  retry_on=(ConnectionError, TimeoutError) # 指定重试的异常类型
  # 注意：LangGraph 1.0的RetryPolicy可能不支持interval、max_attempts等参数
)

@task(retry_policy=api_retry_policy)
def call_external_api(data: str) -> dict:
  """
  模拟调用外部API的任务
  
  Args:
    data: 请求数据
    
  Returns:
    API响应结果
    
  Raises:
    ConnectionError: 模拟网络连接错误
    TimeoutError: 模拟超时错误
  """
  global api_call_attempts
  api_call_attempts += 1
  print(f"调用外部API，第 {api_call_attempts} 次尝试")
  
  # 模拟前两次调用都失败
  if api_call_attempts < 3:
    # 随机抛出不同类型的异常
    error_type = random.choice([ConnectionError, TimeoutError])
    if error_type == ConnectionError:
      print("API调用失败：连接错误")
      raise ConnectionError("无法连接到API服务器")
    else:
      print("API调用失败：请求超时")
      raise TimeoutError("API请求超时")
  
  print("API调用成功")
  return {
    "status": "success",
    "data": f"处理后的数据: {data}",
    "attempts": api_call_attempts
  }

# 数据库操作重试策略示例
db_retry_policy = RetryPolicy(
  retry_on=(Exception,) # 重试所有异常
  # 注意：LangGraph 1.0的RetryPolicy可能不支持interval、max_attempts等参数
)

@task(retry_policy=db_retry_policy)
def database_operation(query: str) -> str:
  """
  模拟数据库操作任务
  
  Args:
    query: 查询语句
    
  Returns:
    查询结果
    
  Raises:
    Exception: 模拟数据库异常
  """
  global database_attempts
  database_attempts += 1
  print(f"执行数据库操作，第 {database_attempts} 次尝试")
  
  # 模拟前几次调用都失败
  if database_attempts < 4:
    print("数据库操作失败：连接超时")
    raise Exception("数据库连接超时")
  
  print("数据库操作成功")
  return f"查询结果: {query} -> 成功"

# 检查点保存器
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def main(inputs: dict) -> dict:
  """
  主工作流
  
  Args:
    inputs: 输入数据字典
    
  Returns:
    处理结果字典
  """
  print("启动主工作流")
  
  # 执行带重试策略的任务
  info_result = get_info().result()
  
  # 执行外部API调用任务
  api_result = call_external_api(inputs.get("data", "默认数据")).result()
  
  # 执行数据库操作任务
  db_result = database_operation(inputs.get("query", "SELECT * FROM users")).result()
  
  return {
    "info": info_result,
    "api_result": api_result,
    "database_result": db_result
  }

# 参考资料中的直接示例
attempts_ref = 0

retry_policy_ref = RetryPolicy(retry_on=ValueError)

@task(retry_policy=retry_policy_ref)
def get_info_ref():
  global attempts_ref
  attempts_ref += 1

  if attempts_ref < 2:
    raise ValueError('Failure')
  return "OK"

def main_demo():
  """主演示函数"""
  print("=== LangGraph 1.0 函数式API重试策略演示 ===")
  
  config = {
    "configurable": {
      "thread_id": "retry-demo-1"
    }
  }
  
  try:
    print("\n--- 执行主工作流 ---")
    result = main.invoke({
      "data": "用户信息", 
      "query": "SELECT * FROM users WHERE id=1"
    }, config=config)
    
    print("\n--- 执行结果 ---")
    print(f"信息获取结果: {result['info']}")
    print(f"API调用结果: {result['api_result']}")
    print(f"数据库操作结果: {result['database_result']}")
    
  except Exception as e:
    print(f"工作流执行失败: {e}")

# 自定义重试条件示例
def should_retry(exception):
  """
  自定义重试条件函数
  
  Args:
    exception: 捕获的异常
    
  Returns:
    bool: 是否应该重试
  """
  # 只有当异常包含特定文本时才重试
  return "temporary" in str(exception).lower()

custom_retry_policy = RetryPolicy(
  retry_on=should_retry  # 使用自定义重试条件
)

@task(retry_policy=custom_retry_policy)
def custom_retry_task(input_data: str) -> str:
  """
  使用自定义重试条件的任务
  
  Args:
    input_data: 输入数据
    
  Returns:
    处理结果
    
  Raises:
    Exception: 模拟异常
  """
  print(f"执行自定义重试任务，输入: {input_data}")
  
  # 模拟临时性错误和永久性错误
  if "temporary" in input_data:
    raise Exception("Temporary network issue")
  elif "permanent" in input_data:
    raise Exception("Permanent failure - should not retry")
  else:
    return "任务执行成功"

@entrypoint(checkpointer=checkpointer)
def custom_retry_workflow(inputs: dict) -> str:
  """
  自定义重试策略工作流
  
  Args:
    inputs: 输入数据
    
  Returns:
    处理结果
  """
  print("启动自定义重试策略工作流")
  result = custom_retry_task(inputs["data"]).result()
  return result

def custom_retry_demo():
  """自定义重试策略演示"""
  print("\n\n=== 自定义重试策略演示 ===")
  
  config = {
    "configurable": {
      "thread_id": "custom-retry-demo"
    }
  }
  
  # 测试临时性错误（应该重试）
  print("\n--- 测试临时性错误（应该重试） ---")
  try:
    result = custom_retry_workflow.invoke({"data": "temporary error"}, config=config)
    print(f"临时性错误处理结果: {result}")
  except Exception as e:
    print(f"临时性错误处理失败: {e}")
  
  # 重置全局状态
  global attempts, api_call_attempts, database_attempts, attempts_ref
  attempts = 0
  api_call_attempts = 0
  database_attempts = 0
  attempts_ref = 0
  
  # 测试永久性错误（不应该重试）
  print("\n--- 测试永久性错误（不应该重试） ---")
  try:
    result = custom_retry_workflow.invoke({"data": "permanent error"}, config=config)
    print(f"永久性错误处理结果: {result}")
  except Exception as e:
    print(f"永久性错误处理失败（符合预期）: {e}")

if __name__ == "__main__":
  main_demo()
  custom_retry_demo()