"""
基于LangGraph 1.0函数式API的示例程序，展示了以下特性：
    1. 使用@task装饰器定义任务节点
     - write_essay: 模拟生成文章的任务
    2. 使用@entrypoint装饰器定义工作流入口
     - 包含中断机制，可以暂停工作流等待外部输入
    3. 中断与恢复机制
     - 使用interrupt()函数暂停工作流并传递数据给外部
     - 使用Command(resume=value)恢复工作流执行
    4. 内存状态保存
     - 使用InMemorySaver()保存工作流状态
示例功能流程：
    1. 启动生成文章任务
    2. 文章生成完成后中断工作流，等待人工审核
    3. 接收人工审核结果
    4. 根据审核结果继续执行并返回最终结果
"""

import time
import uuid

from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

# 模拟写文章的任务节点
@task
def write_essay(topic: str) -> str:
  """根据给定主题写一篇文章"""
  print(f"正在生成关于'{topic}'的文章...")
  time.sleep(1) # 模拟耗时操作
  essay = f"这是一篇关于'{topic}'的文章内容。文章包括引言、正文和结论等部分。"
  print(f"文章生成完成: {essay}")
  return essay

# 定义工作流入口点
@entrypoint(checkpointer=InMemorySaver())
def workflow(topic: str) -> dict:
  """一个简单的工作流，用于生成文章并请求审核"""
  # 执行写文章任务
  essay = write_essay(topic).result()

  # 中断工作流，等待人工审核
  is_approved = interrupt(
    {
      # 提供给中断的JSON序列化数据
      # 当从工作流中流式传输数据时，这些信息会在客户端显示为中断
      "essay": essay, # 待审核的文章
      # 可以添加任何我们需要的额外信息
      # 例如，添加一个名为"action"的键并附上说明指令
      "action": "请审核这篇文章，通过输入True或False来表示是否批准",
    }
  )

  # 返回最终结果
  return {
    "essay": essay, # 生成的文章
    "is_approved": is_approved, # 来自人工审核的响应
  }

def main():
  """主函数，演示如何运行带有人工中断的工作流"""
  print("=== LangGraph 1.0 函数式API 示例 ===")

  # 生成唯一的线程ID用于状态保存
  thread_id = str(uuid.uuid4())
  config = {"configurable": {"thread_id": thread_id}}
  print(f"工作流线程ID: {thread_id}")

  print("\n--- 第一阶段：开始生成文章 ---")
  # 启动工作流并流式输出结果
  stream_iter = workflow.stream("猫", config)

  try:
    for item in stream_iter:
      print(f"流式输出项: {item}")

      # 检查是否有中断信号
      if "__interrupt__" in item:
        print("\n--- 工作流中断，等待人工审核 ---")
        interrupt_data = item["__interrupt__"][0].value
        print(f"待审核文章: {interrupt_data['essay']}")
        print(f"操作提示: {interrupt_data['action']}")

        # 模拟人工审核过程
        print("\n--- 模拟人工审核 ---")
        # 在实际情况中，这里可能会有一个UI界面让用户进行审核
        # 在此示例中，我们直接使用布尔值作为审核结果
        human_review = True # 模拟用户批准

        print(f"审核结果: {'批准' if human_review else '拒绝'}")

        print("\n--- 第二阶段：继续工作流执行 ---")
        # 使用审核结果恢复工作流执行
        for resume_item in workflow.stream(Command(resume=human_review), config):
          print(f"恢复后的流式输出项: {resume_item}")

  except Exception as e:
    print(f"执行过程中出现错误: {e}")

if __name__ == "__main__":
  main()