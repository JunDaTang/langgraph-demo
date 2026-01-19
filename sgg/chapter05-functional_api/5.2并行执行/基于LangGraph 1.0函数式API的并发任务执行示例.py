"""
基于LangGraph 1.0函数式API的并发任务执行示例

本示例演示了如何使用LangGraph的Functional API并行执行多个任务，
特别适用于IO密集型任务（如调用大语言模型API）以提高性能。

主要特性：
1. 使用@task装饰器定义可并行执行的任务
2. 使用@entrypoint装饰器定义工作流入口点
3. 并发执行多个任务并等待所有结果
4. 使用InMemorySaver进行状态持久化
"""

import time
import uuid
from typing import List

# 导入LangGraph Functional API相关组件
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

# 模拟耗时的IO操作（如调用LLM API）
def simulate_llm_call(topic: str, delay: float = 1.0) -> str:
  """
  模拟调用大语言模型API生成段落
  
  Args:
    topic: 主题
    delay: 模拟延迟时间（秒）
  
  Returns:
    生成的段落内容
  """
  print(f"开始生成关于'{topic}'的段落...")
  time.sleep(delay) # 模拟网络IO延迟
  paragraph = f"这是关于'{topic}'的一段内容。在这个段落中，我们会讨论这个主题的各种方面和细节。"
  print(f"完成'{topic}'段落生成")
  return paragraph

# 定义任务函数 - 生成指定主题的段落
@task
def generate_paragraph(topic: str) -> str:
  """
  生成关于给定主题的段落
  
  Args:
    topic: 段落主题
  
  Returns:
    生成的段落内容
  """
  return simulate_llm_call(topic, 1.0)

# 定义工作流入口点
@entrypoint(checkpointer=InMemorySaver())
def workflow(topics: List[str]) -> str:
  """
  并行生成多个主题的段落并组合成完整文本
  
  Args:
    topics: 主题列表
  
  Returns:
    组合后的完整文本
  """
  # 并行启动所有任务
  futures = [generate_paragraph(topic) for topic in topics]
  
  # 等待所有任务完成并获取结果
  paragraphs = [f.result() for f in futures]
  
  # 用双换行符合并所有段落
  return "\n\n".join(paragraphs)

def main():
  """主函数，演示并发执行任务"""
  print("=== LangGraph 1.0 函数式API 并发执行示例 ===")
  
  # 定义要处理的主题列表
  topics = ["量子计算", "气候变化", "航空史"]
  print(f"待处理的主题: {topics}")
  
  # 生成唯一线程ID用于状态保存
  thread_id = str(uuid.uuid4())
  config = {"configurable": {"thread_id": thread_id}}
  print(f"工作流线程ID: {thread_id}")
  
  # 记录开始时间
  start_time = time.time()
  
  # 执行工作流
  print("\n--- 开始并行执行任务 ---")
  result = workflow.invoke(topics, config=config)
  
  # 计算执行时间
  elapsed_time = time.time() - start_time
  
  # 输出结果
  print("\n--- 生成结果 ---")
  print(result)
  
  print(f"\n--- 执行统计 ---")
  print(f"处理主题数量: {len(topics)} 个")
  print(f"总耗时: {elapsed_time:.2f} 秒")
  print(f"平均每个主题耗时: {elapsed_time/len(topics):.2f} 秒")
  
  # 对比串行执行
  print("\n--- 串行执行对比 ---")
  serial_start = time.time()
  serial_results = []
  for topic in topics:
    serial_results.append(simulate_llm_call(topic, 1.0))
  serial_elapsed = time.time() - serial_start
  print(f"串行执行总耗时: {serial_elapsed:.2f} 秒")
  
  print(f"\n--- 性能提升 ---")
  improvement = serial_elapsed / elapsed_time
  print(f"并行执行相较于串行执行提升了 {improvement:.1f} 倍性能")

if __name__ == "__main__":
  main()