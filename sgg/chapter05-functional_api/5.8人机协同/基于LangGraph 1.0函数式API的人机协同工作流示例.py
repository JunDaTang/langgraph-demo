"""
基于LangGraph 1.0函数式API的人机协同工作流示例

本示例演示了如何在LangGraph函数式API中实现人机协同工作流，
展示了使用interrupt函数和Command原语实现人机交互的机制。

主要特性：
1. 使用interrupt函数实现人机交互中断
2. 使用Command原语恢复工作流执行
3. 使用InMemorySaver进行状态持久化
4. 实现任务间的顺序执行和人机协同
"""

from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

@task
def step_1(input_query):
  """
  第一步任务：追加"bar"
  
  参数:
    input_query: 输入查询字符串
    
  返回:
    追加"bar"后的字符串
  """
  print(f"执行第一步任务，输入: {input_query}")
  result = f"{input_query} bar"
  print(f"第一步任务完成，结果: {result}")
  return result

@task
def human_feedback(input_query):
  """
  人工反馈任务：暂停以等待人工输入，恢复时附加人工输入
  
  参数:
    input_query: 输入查询字符串
    
  返回:
    追加人工反馈后的字符串
  """
  print(f"执行人工反馈任务，输入: {input_query}")
  # 中断执行，等待人工输入
  feedback = interrupt(f"请提供反馈: {input_query}")
  result = f"{input_query} {feedback}"
  print(f"人工反馈任务完成，结果: {result}")
  return result

@task
def step_3(input_query):
  """
  第三步任务：追加"qux"
  
  参数:
    input_query: 输入查询字符串
    
  返回:
    追加"qux"后的字符串
  """
  print(f"执行第三步任务，输入: {input_query}")
  result = f"{input_query} qux"
  print(f"第三步任务完成，结果: {result}")
  return result

# 初始化检查点保存器
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def graph(input_query):
  """
  主工作流：组合三个任务的执行
  
  参数:
    input_query: 输入查询字符串
    
  返回:
    最终处理结果
  """
  print("启动主工作流")
  
  # 顺序执行三个任务
  result_1 = step_1(input_query).result()
  result_2 = human_feedback(result_1).result()
  result_3 = step_3(result_2).result()

  return result_3

def main_demo():
  """主演示函数"""
  print("=== LangGraph 1.0 函数式API人机协同工作流演示 ===")
  
  # 配置工作流执行参数
  config = {"configurable": {"thread_id": "human-collaboration-demo-1"}}
  
  print("\n--- 启动工作流执行 ---")
  # 启动工作流执行
  for event in graph.stream("foo", config):
    print(f"工作流事件: {event}")
    print()
  
  print("--- 工作流在human_feedback任务处中断，等待人工输入 ---")
  
  print("\n--- 恢复工作流执行，提供人工输入'baz' ---")
  # 继续执行工作流，提供人工输入
  for event in graph.stream(Command(resume="baz"), config):
    print(f"工作流事件: {event}")
    print()
  
  print("--- 工作流执行完成 ---")

# 更复杂的人机协同示例
@task
def data_processing(input_data):
  """
  数据处理任务
  
  参数:
    input_data: 输入数据
    
  返回:
    处理后的数据
  """
  print(f"执行数据处理任务，输入: {input_data}")
  processed_data = f"已处理数据: {input_data}"
  print(f"数据处理完成: {processed_data}")
  return processed_data

@task
def approval_process(input_data):
  """
  审批流程任务：等待人工审批
  
  参数:
    input_data: 待审批的数据
    
  返回:
    审批结果
  """
  print(f"执行审批流程任务，待审批数据: {input_data}")
  # 等待人工审批
  approval_result = interrupt(f"请审批以下数据:\n{input_data}\n输入'approved'批准或'rejected'拒绝:")
  
  if approval_result.lower() == "approved":
    result = f"{input_data} [已批准]"
    print(f"审批通过: {result}")
  elif approval_result.lower() == "rejected":
    result = f"{input_data} [已拒绝]"
    print(f"审批拒绝: {result}")
  else:
    result = f"{input_data} [审批结果未知: {approval_result}]"
    print(f"未知审批结果: {result}")
    
  return result

@task
def finalization(input_data):
  """
  最终化任务
  
  参数:
    input_data: 输入数据
    
  返回:
    最终结果
  """
  print(f"执行最终化任务，输入: {input_data}")
  result = f"最终结果: {input_data}"
  print(f"最终化完成: {result}")
  return result

complex_checkpointer = InMemorySaver()

@entrypoint(checkpointer=complex_checkpointer)
def complex_workflow(initial_data):
  """
  复杂人机协同工作流
  
  参数:
    initial_data: 初始数据
    
  返回:
    最终处理结果
  """
  print("启动复杂人机协同工作流")
  
  # 顺序执行任务
  processed_data = data_processing(initial_data).result()
  approved_data = approval_process(processed_data).result()
  final_result = finalization(approved_data).result()
  
  return final_result

def complex_demo():
  """复杂人机协同演示"""
  print("\n\n=== 复杂人机协同演示 ===")
  
  config = {"configurable": {"thread_id": "complex-human-collaboration-demo"}}
  
  print("\n--- 启动复杂工作流执行 ---")
  # 启动工作流执行
  for event in complex_workflow.stream("原始数据", config):
    print(f"工作流事件: {event}")
    print()
  
  print("--- 工作流在审批流程处中断，等待人工审批 ---")
  
  print("\n--- 恢复工作流执行，提供审批输入'approved' ---")
  # 继续执行工作流，提供审批结果
  for event in complex_workflow.stream(Command(resume="approved"), config):
    print(f"工作流事件: {event}")
    print()
  
  print("--- 复杂工作流执行完成 ---")

# 多轮人机交互示例
@task
def initial_task(input_data):
  """初始任务"""
  print(f"执行初始任务: {input_data}")
  return f"初始处理: {input_data}"

@task
def first_interrupt(input_data):
  """第一次人工交互"""
  print(f"第一次人工交互任务: {input_data}")
  feedback = interrupt(f"第一次交互，请提供反馈 [{input_data}]:")
  return f"{input_data} + 反馈1[{feedback}]"

@task
def middle_task(input_data):
  """中间任务"""
  print(f"执行中间任务: {input_data}")
  return f"中间处理: {input_data}"

@task
def second_interrupt(input_data):
  """第二次人工交互"""
  print(f"第二次人工交互任务: {input_data}")
  feedback = interrupt(f"第二次交互，请提供反馈 [{input_data}]:")
  return f"{input_data} + 反馈2[{feedback}]"

@task
def final_task(input_data):
  """最终任务"""
  print(f"执行最终任务: {input_data}")
  return f"最终结果: {input_data}"

multi_checkpointer = InMemorySaver()

@entrypoint(checkpointer=multi_checkpointer)
def multi_interrupt_workflow(initial_data):
  """
  多轮人机交互工作流
  
  参数:
    initial_data: 初始数据
    
  返回:
    最终处理结果
  """
  print("启动多轮人机交互工作流")
  
  # 顺序执行任务
  result1 = initial_task(initial_data).result()
  result2 = first_interrupt(result1).result()
  result3 = middle_task(result2).result()
  result4 = second_interrupt(result3).result()
  result5 = final_task(result4).result()
  
  return result5

def multi_interrupt_demo():
  """多轮人机交互演示"""
  print("\n\n=== 多轮人机交互演示 ===")
  
  config = {"configurable": {"thread_id": "multi-interrupt-demo"}}
  
  print("\n--- 启动多轮交互工作流执行 ---")
  # 启动工作流执行
  for event in multi_interrupt_workflow.stream("开始数据", config):
    print(f"工作流事件: {event}")
    print()
  
  print("--- 工作流在第一次交互处中断 ---")
  
  print("\n--- 恢复工作流执行，提供第一次反馈 ---")
  # 第一次恢复执行
  for event in multi_interrupt_workflow.stream(Command(resume="第一次反馈内容"), config):
    print(f"工作流事件: {event}")
    print()
  
  print("--- 工作流在第二次交互处中断 ---")
  
  print("\n--- 恢复工作流执行，提供第二次反馈 ---")
  # 第二次恢复执行
  for event in multi_interrupt_workflow.stream(Command(resume="第二次反馈内容"), config):
    print(f"工作流事件: {event}")
    print()
  
  print("--- 多轮交互工作流执行完成 ---")

if __name__ == "__main__":
  # 执行所有演示
  main_demo()
  complex_demo()
  multi_interrupt_demo()