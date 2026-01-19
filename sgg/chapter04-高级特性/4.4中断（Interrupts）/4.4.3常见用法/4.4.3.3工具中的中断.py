"""
LangGraph 工具中断演示

该演示展示了如何在工具函数内部使用中断功能，
使工具在每次被调用时暂停以等待批准，并允许在执行前进行人工检查和编辑。
"""

import sqlite3
from typing import TypedDict

from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

class AgentState(TypedDict):
  """代理状态定义"""
  messages: list[dict]

@tool
def send_email(to: str, subject: str, body: str):
  """
  发送邮件给收件人。
  
  Args:
    to (str): 收件人邮箱地址
    subject (str): 邮件主题
    body (str): 邮件正文
    
  Returns:
    str: 发送结果信息
  """
  print(f"执行工具: send_email")
  print(f"收件人: {to}")
  print(f"主题: {subject}")
  print(f"正文: {body}")
  
  # 在发送前暂停；有效载荷会出现在 result["__interrupt__"] 中
  response = interrupt({
    "action": "send_email",
    "to": to,
    "subject": subject,
    "body": body,
    "message": "是否批准发送此邮件？",
  })

  if response.get("action") == "approve":
    final_to = response.get("to", to)
    final_subject = response.get("subject", subject)
    final_body = response.get("body", body)

    # 实际发送邮件（这里只是模拟）
    print(f"[send_email] to={final_to} subject={final_subject} body={final_body}")
    return f"邮件已发送至 {final_to}"

  return "用户取消了邮件发送"

def agent_node(state: AgentState):
  """
  代理节点函数
  
  Args:
    state: 当前状态，包含消息历史
    
  Returns:
    dict: 更新后的状态
  """
  print("执行节点: agent_node")
  
  # 模拟LLM决定调用工具
  # 在实际应用中，这里会使用LLM来决定是否调用工具
  if len(state["messages"]) == 1: # 第一次调用
    # 模拟LLM决定调用send_email工具
    tool_call = {
      "name": "send_email",
      "arguments": {
        "to": "alice@example.com",
        "subject": "会议安排",
        "body": "你好，我想安排一个会议讨论项目进展。"
      }
    }
    
    # 调用工具（这会触发中断）
    try:
      result = send_email.invoke(tool_call["arguments"])
      return {
        "messages": state["messages"] + [
          {"role": "assistant", "content": f"调用工具: {tool_call['name']}"},
          {"role": "tool", "name": tool_call["name"], "content": result}
        ]
      }
    except Exception as e:
      # 捕获中断异常，让工作流暂停
      raise e
  else:
    # 后续调用，返回最终结果
    return {"messages": state["messages"]}

def main():
  """主函数 - 演示工具中断功能"""
  print("=== LangGraph 工具中断演示 ===\n")
  
  # 创建状态图
  builder = StateGraph(AgentState)
  builder.add_node("agent", agent_node)
  builder.add_edge(START, "agent")
  builder.add_edge("agent", END)

  # 使用内存保存器作为检查点
  checkpointer = MemorySaver()
  
  # 编译图
  graph = builder.compile(checkpointer=checkpointer)

  # 配置线程ID
  config = {"configurable": {"thread_id": "email-workflow"}}
  
  # 初始化状态并执行图
  print("1. 启动邮件发送工作流...")
  try:
    initial = graph.invoke(
      {
        "messages": [
          {"role": "user", "content": "请发送邮件给alice@example.com关于会议安排"}
        ]
      },
      config=config,
    )
    print(f"工作流中断信息: {initial['__interrupt__']}\n")
    
    # 模拟用户审批过程
    print("2. 模拟用户审批过程...")
    interrupt_value = initial["__interrupt__"][0].value
    print("操作:", interrupt_value["action"])
    print("消息:", interrupt_value["message"])
    print("收件人:", interrupt_value["to"])
    print("主题:", interrupt_value["subject"])
    print("正文:", interrupt_value["body"])
    
    # 获取用户输入
    while True:
      user_input = input("\n是否批准发送邮件？(y/n): ").strip().lower()
      if user_input in ['y', 'yes', '是']:
        # 用户批准，可以编辑参数
        new_subject = input("请输入新主题（直接回车保持原主题）: ").strip()
        if not new_subject:
          approval_response = {"action": "approve"}
        else:
          approval_response = {"action": "approve", "subject": new_subject}
        break
      elif user_input in ['n', 'no', '否']:
        approval_response = {"action": "reject"}
        break
      else:
        print("无效输入，请输入 y/yes/是 或 n/no/否")
    
    # 使用用户决定恢复执行
    print(f"\n3. 使用审批决定恢复工作流执行...")
    resumed = graph.invoke(
      Command(resume=approval_response),
      config=config,
    )
    
    # 显示最终结果
    print(f"最终消息: {resumed['messages'][-1]}")
    print("\n=== 演示完成 ===")
    
  except Exception as e:
    print(f"执行过程中出现错误: {e}")
    print("\n=== 演示结束 ===")

if __name__ == "__main__":
  main()


# === LangGraph 工具中断演示 ===

# 1. 启动邮件发送工作流...
# 执行节点: agent_node
# 执行工具: send_email
# 收件人: alice@example.com
# 主题: 会议安排
# 正文: 你好，我想安排一个会议讨论项目进展。
# 工作流中断信息: [Interrupt(value={'action': 'send_email', 'to': 'alice@example.com', 'subject': '会议安排', 'body': '你好，我想 
# 安排一个会议讨论项目进展。', 'message': '是否批准发送此邮件？'}, id='6e8be4e24dbf1f87f1aad159b9b09bcd')]

# 2. 模拟用户审批过程...
# 操作: send_email
# 消息: 是否批准发送此邮件？
# 收件人: alice@example.com
# 主题: 会议安排
# 正文: 你好，我想安排一个会议讨论项目进展。

# 是否批准发送邮件？(y/n): y
# 请输入新主题（直接回车保持原主题）: 2026如何发财

# 3. 使用审批决定恢复工作流执行...
# 执行节点: agent_node
# 执行工具: send_email
# 收件人: alice@example.com
# 主题: 会议安排
# 正文: 你好，我想安排一个会议讨论项目进展。
# [send_email] to=alice@example.com subject=2026如何发财 body=你好，我想安排一个会议讨论项目进展。
# 最终消息: {'role': 'tool', 'name': 'send_email', 'content': '邮件已发送至 alice@example.com'}

# === 演示完成 ===