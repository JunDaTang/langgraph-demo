"""
LangGraph 审阅和编辑工作流演示

该演示展示了如何使用 LangGraph 的中断功能实现人工审阅和编辑工作流。
这对于让人类在继续之前审核并编辑图状态非常有用。
"""

import sqlite3
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

class ReviewState(TypedDict):
  """审阅状态定义"""
  generated_text: str

def review_node(state: ReviewState):
  """
  审阅节点
  
  Args:
    state: 当前状态，包含生成的文本内容
    
  Returns:
    dict: 更新后的状态
  """
  print(f"执行节点: review_node")
  print(f"当前文本内容: {state['generated_text']}")
  print("工作流暂停，等待用户审阅和编辑...")
  
  # 请求审阅者编辑生成的内容
  updated = interrupt({
    "instruction": "请审阅并编辑以下内容",
    "content": state["generated_text"],
  })
  
  print(f"收到编辑后的内容: {updated}")
  return {"generated_text": updated}

def main():
  """主函数 - 演示审阅和编辑工作流"""
  print("=== LangGraph 审阅和编辑工作流演示 ===\n")
  
  # 创建状态图
  builder = StateGraph(ReviewState)
  builder.add_node("review", review_node)
  builder.add_edge(START, "review")
  builder.add_edge("review", END)

  # 使用内存保存器作为检查点
  checkpointer = MemorySaver()
  
  # 编译图
  graph = builder.compile(checkpointer=checkpointer)

  # 配置线程ID
  config = {"configurable": {"thread_id": "review-42"}}
  
  # 初始化状态并执行图
  print("1. 启动审阅工作流...")
  initial = graph.invoke({"generated_text": "这是初始草稿内容"}, config=config)
  
  # 显示中断信息
  print(f"工作流中断信息: {initial['__interrupt__']}\n")
  
  # 模拟用户审阅和编辑过程
  print("2. 模拟用户审阅和编辑过程...")
  interrupt_value = initial["__interrupt__"][0].value
  print("指导说明:", interrupt_value["instruction"])
  print("原文内容:", interrupt_value["content"])
  
  # 获取用户编辑后的内容
  edited_text = input("\n请输入编辑后的内容: ").strip()
  
  # 使用用户编辑后的内容恢复执行
  print(f"\n3. 使用编辑后的内容恢复工作流执行...")
  final_state = graph.invoke(
    Command(resume=edited_text),
    config=config,
  )
  
  # 显示最终结果
  print(f"最终状态: {final_state}")
  print(f"最终文本内容: {final_state['generated_text']}")
  print("\n=== 演示完成 ===")

if __name__ == "__main__":
  main()


# === LangGraph 审阅和编辑工作流演示 ===

# 1. 启动审阅工作流...
# 执行节点: review_node
# 当前文本内容: 这是初始草稿内容
# 工作流暂停，等待用户审阅和编辑...
# 工作流中断信息: [Interrupt(value={'instruction': '请审阅并编辑以下内容', 'content': '这是初始草稿内容'}, id='5d71dc21bc2f1a9c347ef10697d7d79b')]

# 2. 模拟用户审阅和编辑过程...
# 指导说明: 请审阅并编辑以下内容
# 原文内容: 这是初始草稿内容

# 请输入编辑后的内容: 这是复核草稿内容   

# 3. 使用编辑后的内容恢复工作流执行...
# 执行节点: review_node
# 当前文本内容: 这是初始草稿内容
# 工作流暂停，等待用户审阅和编辑...
# 收到编辑后的内容: 这是复核草稿内容
# 最终状态: {'generated_text': '这是复核草稿内容'}
# 最终文本内容: 这是复核草稿内容

# === 演示完成 ===