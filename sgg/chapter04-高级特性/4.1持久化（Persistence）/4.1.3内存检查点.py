"""
LangGraph 1.0 持久化存储演示 - 内存存储 (In-Memory)

特点：
- 数据暂存于内存，程序关闭后丢失
- 无需额外配置
- 适用于本地测试和临时验证工作流逻辑
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import operator

# 定义状态
class PersistenceDemoState(TypedDict):
  messages: Annotated[list, operator.add]
  step_count: Annotated[int, operator.add]

# 节点函数
def step_one(state: PersistenceDemoState) -> dict:
  print("执行步骤 1")
  return {
    "messages": ["执行了步骤 1"],
    "step_count": 1
  }

def step_two(state: PersistenceDemoState) -> dict:
  print("执行步骤 2")
  return {
    "messages": ["执行了步骤 2"],
    "step_count": 1
  }

def step_three(state: PersistenceDemoState) -> dict:
  print("执行步骤 3")
  return {
    "messages": ["执行了步骤 3"],
    "step_count": 1
  }

# 构建图
def create_graph():
  builder = StateGraph(PersistenceDemoState)
  builder.add_node("step_one", step_one)
  builder.add_node("step_two", step_two)
  builder.add_node("step_three", step_three)

  builder.add_edge(START, "step_one")
  builder.add_edge("step_one", "step_two")
  builder.add_edge("step_two", "step_three")
  builder.add_edge("step_three", END)

  return builder

def main():
  print("=== LangGraph 1.0 内存持久化存储演示 ===\n")

  # 创建内存存储器
  memory = InMemorySaver()
  
  # 编译图并使用内存存储
  graph = create_graph()
  app = graph.compile(checkpointer=memory)
  
  # 配置线程ID用于存储状态
  config = {"configurable": {"thread_id": "demo_thread_1"}}
  
  print("1. 首次执行工作流:")
  result = app.invoke({
    "messages": ["开始执行"],
    "step_count": 0
  }, config)
  
  print(f"执行结果: {result}\n")
  
  print("2. 检查存储的状态:")
  saved_state = app.get_state(config)
  print(f"保存的状态: {saved_state.values}")
  print(f"下一个节点: {saved_state.next}\n")
  
  print("3. 恢复执行工作流:")
  # 由于工作流已经完成，这里会直接返回最终结果
  result2 = app.invoke(None, config)
  print(f"恢复执行结果: {result2}\n")
  
  print("=== 演示结束 ===")

if __name__ == "__main__":
  main()



# === LangGraph 1.0 内存持久化存储演示 ===

# 1. 首次执行工作流:
# 执行步骤 1
# 执行步骤 2
# 执行步骤 3
# 执行结果: {'messages': ['开始执行', '执行了步骤 1', '执行了步骤 2', '执行了步骤 3'], 'step_count': 3}

# 2. 检查存储的状态:
# 保存的状态: {'messages': ['开始执行', '执行了步骤 1', '执行了步骤 2', '执行了步骤 3'], 'step_count': 3}
# 下一个节点: ()

# 3. 恢复执行工作流:
# 恢复执行结果: {'messages': ['开始执行', '执行了步骤 1', '执行了步骤 2', '执行了步骤 3'], 'step_count': 3}

# === 演示结束 ===