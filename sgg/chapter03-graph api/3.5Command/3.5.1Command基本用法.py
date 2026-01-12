"""
LangGraph Command 基础演示

演示如何在节点中使用 Command 对象同时更新状态和控制流程。
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# 定义状态
class AgentState(TypedDict):
  messages: Annotated[list, lambda x, y: x + y]
  current_agent: str
  task_completed: bool

# 节点函数：决策代理
def decision_agent(state: AgentState) -> Command[AgentState]:
  """决策代理节点，根据消息内容决定下一步操作"""
  print("执行节点: decision_agent")
  
  # 检查最新的消息
  last_message = state["messages"][-1] if state["messages"] else ""
  print(f"最新消息: {last_message}")
  
  # 根据消息内容决定下一步
  if "数学" in last_message:
    # 更新状态并跳转到数学代理
    return Command(
      update={
        "messages": [("system", "路由到数学代理")],
        "current_agent": "math_agent"
      },
      goto="math_agent"
    )
  elif "翻译" in last_message:
    # 更新状态并跳转到翻译代理
    return Command(
      update={
        "messages": [("system", "路由到翻译代理")],
        "current_agent": "translation_agent"
      },
      goto="translation_agent"
    )
  else:
    # 任务完成
    return Command(
      update={
        "messages": [("system", "任务完成")],
        "task_completed": True
      },
      goto=END
    )

# 节点函数：数学代理
def math_agent(state: AgentState) -> Command[AgentState]:
  """数学代理节点"""
  print("执行节点: math_agent")
  
  # 执行数学计算任务
  result = "2 + 2 = 4"
  print(f"计算结果: {result}")
  
  # 更新状态并返回决策代理
  return Command(
    update={
      "messages": [("assistant", f"数学计算结果: {result}")],
      "current_agent": "decision_agent"
    },
    goto="decision_agent"
  )

# 节点函数：翻译代理
def translation_agent(state: AgentState) -> Command[AgentState]:
  """翻译代理节点"""
  print("执行节点: translation_agent")
  
  # 执行翻译任务
  translation = "Hello -> 你好"
  print(f"翻译结果: {translation}")
  
  # 更新状态并返回决策代理
  return Command(
    update={
      "messages": [("assistant", f"翻译结果: {translation}")],
      "current_agent": "decision_agent"
    },
    goto="decision_agent"
  )

def main():
  """演示Command基础用法"""
  print("=== Command 基础演示 ===\n")
  
  # 创建图
  builder = StateGraph(AgentState)
  
  # 添加节点
  builder.add_node("decision_agent", decision_agent)
  builder.add_node("math_agent", math_agent)
  builder.add_node("translation_agent", translation_agent)
  
  # 设置入口点
  builder.add_edge(START, "decision_agent")
  
  # 编译图
  graph = builder.compile()
  
  # 执行图 - 测试数学任务
  print("测试1: 数学任务")
  initial_state = {
    "messages": [("user", "我需要计算数学题")],
    "current_agent": "user",
    "task_completed": False
  }
  print("初始状态:", initial_state)
  result = graph.invoke(initial_state)
  print("最终状态:", result)
  print("\n" + "="*50 + "\n")
  
  # 执行图 - 测试翻译任务
  print("测试2: 翻译任务")
  initial_state = {
    "messages": [("user", "我需要翻译文本")],
    "current_agent": "user",
    "task_completed": False
  }
  print("初始状态:", initial_state)
  result = graph.invoke(initial_state)
  print("最终状态:", result)
  print("\n" + "="*50 + "\n")
  
  # 执行图 - 测试完成任务
  print("测试3: 完成任务")
  initial_state = {
    "messages": [("user", "你好")],
    "current_agent": "user",
    "task_completed": False
  }
  print("初始状态:", initial_state)
  result = graph.invoke(initial_state)
  print("最终状态:", result)

if __name__ == "__main__":
  main()