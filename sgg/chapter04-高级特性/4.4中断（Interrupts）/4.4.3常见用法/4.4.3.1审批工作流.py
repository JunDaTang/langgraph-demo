"""
LangGraph 审批工作流演示

该演示展示了如何使用 LangGraph 的中断功能实现需要人工审批的工作流。
当工作流遇到关键操作时会暂停，并等待用户的批准或拒绝。
"""

import sqlite3
from typing import Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

class ApprovalState(TypedDict):
  """审批状态定义"""
  action_details: str
  status: Optional[Literal["pending", "approved", "rejected"]]

def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
  """
  审批节点
  
  Args:
    state: 当前状态
    
  Returns:
    Command: 包含中断信息和后续路由指令的命令对象
  """
  print(f"执行节点: approval_node")
  print(f"操作详情: {state['action_details']}")
  print("工作流暂停，等待用户审批...")
  
  # 中断执行并暴露详细信息供调用方在UI中渲染
  decision = interrupt({
    "question": "批准此操作吗？",
    "details": state["action_details"],
  })
  
  # 根据恢复值路由到适当的节点
  next_node = "proceed" if decision else "cancel"
  print(f"审批决定: {'批准' if decision else '拒绝'}，路由到节点: {next_node}")
  
  return Command(goto=next_node)

def proceed_node(state: ApprovalState):
  """
  执行节点 - 当审批被批准时执行
  
  Args:
    state: 当前状态
    
  Returns:
    dict: 更新后的状态
  """
  print("执行节点: proceed_node")
  print("操作已被批准，正在执行...")
  return {"status": "approved"}

def cancel_node(state: ApprovalState):
  """
  取消节点 - 当审批被拒绝时执行
  
  Args:
    state: 当前状态
    
  Returns:
    dict: 更新后的状态
  """
  print("执行节点: cancel_node")
  print("操作已被拒绝，正在取消...")
  return {"status": "rejected"}

def main():
  """主函数 - 演示审批工作流"""
  print("=== LangGraph 审批工作流演示 ===\n")
  
  # 创建状态图
  builder = StateGraph(ApprovalState)
  builder.add_node("approval", approval_node)
  builder.add_node("proceed", proceed_node)
  builder.add_node("cancel", cancel_node)
  builder.add_edge(START, "approval")
  
  # 注意：这里我们不直接连接 approval 到 proceed 和 cancel
  # 而是通过 Command(goto=...) 在 approval_node 中动态决定
  
  builder.add_edge("proceed", END)
  builder.add_edge("cancel", END)

  # 使用内存保存器作为检查点
  checkpointer = MemorySaver()
  
  # 编译图
  graph = builder.compile(checkpointer=checkpointer)

  # 配置线程ID
  config = {"configurable": {"thread_id": "approval-123"}}
  
  # 初始化状态并执行图
  print("1. 启动审批工作流...")
  initial = graph.invoke(
    {"action_details": "转账 $500", "status": "pending"},
    config=config,
  )
  
  # 显示中断信息
  print(f"工作流中断信息: {initial['__interrupt__']}\n")
  
  # 模拟用户审批过程
  print("2. 模拟用户审批过程...")
  interrupt_value = initial["__interrupt__"][0].value
  print("操作详情:", interrupt_value["details"])
  print("问题:", interrupt_value["question"])
  
  # 获取用户输入
  while True:
    user_input = input("\n请输入审批决定 (y/n): ").strip().lower()
    if user_input in ['y', 'yes', '是']:
      decision = True
      break
    elif user_input in ['n', 'no', '否']:
      decision = False
      break
    else:
      print("无效输入，请输入 y/yes/是 或 n/no/否")
  
  # 使用用户决定恢复执行
  print(f"\n3. 使用审批决定恢复工作流执行...")
  resumed = graph.invoke(Command(resume=decision), config=config)
  
  # 显示最终结果
  print(f"最终状态: {resumed}")
  print(f"操作状态: {resumed['status']}")
  print("\n=== 演示完成 ===")

if __name__ == "__main__":
  main()



# === LangGraph 审批工作流演示 ===

# 1. 启动审批工作流...
# 执行节点: approval_node
# 操作详情: 转账 $500
# 工作流暂停，等待用户审批...
# 工作流中断信息: [Interrupt(value={'question': '批准此操作吗？', 'details': '转账 $500'}, id='2b35700814cdedd82f4ad8986544c3fc')]
# 2. 模拟用户审批过程...
# 操作详情: 转账 $500
# 问题: 批准此操作吗？

# 请输入审批决定 (y/n): y

# 3. 使用审批决定恢复工作流执行...
# 执行节点: approval_node
# 操作详情: 转账 $500
# 工作流暂停，等待用户审批...
# 审批决定: 批准，路由到节点: proceed
# 执行节点: proceed_node
# 操作已被批准，正在执行...
# 最终状态: {'action_details': '转账 $500', 'status': 'approved'}
# 操作状态: approved

# === 演示完成 ===

# (langgraph-demo) E:\BaiduSyncdisk\github\langgraph-demo>python "sgg\chapter04-高级特性\4.4中断（Interrupts）\4.4.3常见用法\4.4.3.1审批工作流.py"
# === LangGraph 审批工作流演示 ===

# 1. 启动审批工作流...
# 执行节点: approval_node
# 操作详情: 转账 $500
# 工作流暂停，等待用户审批...
# 工作流中断信息: [Interrupt(value={'question': '批准此操作吗？', 'details': '转账 $500'}, id='27304e89c1a32b68588beb7833444ed5')]
# 2. 模拟用户审批过程...
# 操作详情: 转账 $500
# 问题: 批准此操作吗？

# 请输入审批决定 (y/n): n

# 3. 使用审批决定恢复工作流执行...
# 执行节点: approval_node
# 操作详情: 转账 $500
# 工作流暂停，等待用户审批...
# 审批决定: 拒绝，路由到节点: cancel
# 执行节点: cancel_node
# 操作已被拒绝，正在取消...
# 最终状态: {'action_details': '转账 $500', 'status': 'rejected'}
# 操作状态: rejected

# === 演示完成 ===