"""
基于LangGraph 1.0函数式API的副作用处理简单示例

这个示例更展示了如何正确封装副作用。
"""

from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

# 正确的方式：将副作用封装在task中
@task
def write_to_file(): 
  with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Side effect executed")
  return "文件写入完成"

@entrypoint(checkpointer=InMemorySaver())
def my_workflow(inputs: dict) -> dict:
  # 正确的方式：副作用被封装在任务中
  result = write_to_file().result()
  value = interrupt("question")
  return {
    "file_result": result,
    "user_response": value
  }

# 错误的方式：直接在工作流中执行副作用
@entrypoint(checkpointer=InMemorySaver())
def my_incorrect_workflow(inputs: dict) -> dict:
  # 错误的方式：副作用直接包含在工作流中
  # 当恢复工作流时，会再次执行这个副作用
  with open("output_incorrect.txt", "w", encoding="utf-8") as f: 
    f.write("Side effect executed") 
  value = interrupt("question")
  return {
    "warning": "副作用会重复执行",
    "user_response": value
  }

def demo():
  """演示正确和不正确的副作用处理方式"""
  print("=== 副作用处理演示 ===\n")
  
  # 正确方式演示
  print("1. 正确的副作用处理方式:")
  config = {"configurable": {"thread_id": "correct-demo"}}
  
  for event in my_workflow.stream({"input": "test"}, config):
    print(f"  事件: {event}")
  
  # 恢复执行
  for event in my_workflow.stream(Command(resume="answer"), config):
    print(f"  恢复事件: {event}")
  
  print("\n2. 不正确的副作用处理方式:")
  config2 = {"configurable": {"thread_id": "incorrect-demo"}}
  
  for event in my_incorrect_workflow.stream({"input": "test"}, config2):
    print(f"  事件: {event}")
  
  # 恢复执行（副作用会重复执行）
  for event in my_incorrect_workflow.stream(Command(resume="answer"), config2):
    print(f"  恢复事件: {event}")
    print("  注意：文件会被重复写入！")

if __name__ == "__main__":
  demo()