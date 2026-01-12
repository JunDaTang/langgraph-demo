"""
LangGraph条件入口点演示

条件入口点允许根据输入状态动态决定从哪个节点开始执行。
"""

from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 定义状态
class GraphState(TypedDict):
  value: int
  step: str

# 定义节点函数
def node_a(state: GraphState) -> dict:
  """节点A"""
  print("执行节点A")
  return {"value": state["value"] + 1, "step": "A执行完毕"}

def node_b(state: GraphState) -> dict:
  """节点B"""
  print("执行节点B")
  return {"value": state["value"] * 2, "step": "B执行完毕"}

def node_d(state: GraphState) -> dict:
  """节点D"""
  print("执行节点D")
  return {"value": state["value"] + 10, "step": "D执行完毕"}

# 条件入口点的路由函数
def entry_condition(state: GraphState) -> Literal["node_a", "node_d"]:
  """根据输入值决定从哪个节点开始"""
  if state.get("value", 0) > 5:
    return "node_d" # 大于5从节点D开始
  else:
    return "node_a" # 否则从节点A开始

def main():
  """演示条件入口点"""
  print("=== 条件入口点演示 ===")
  
  # 创建图
  builder = StateGraph(GraphState)
  
  # 添加节点
  builder.add_node("node_a", node_a)
  builder.add_node("node_d", node_d)
  builder.add_node("node_b", node_b)
  
  # 添加条件入口点
  builder.add_conditional_edges(
    START, # 起始点
    entry_condition, # 路由函数
    { # 路由映射
      "node_a": "node_a",
      "node_d": "node_d"
    }
  )
  
  # 添加普通边
  builder.add_edge("node_a", "node_b")
  builder.add_edge("node_d", "node_b")
  builder.add_edge("node_b", END)
  
  # 编译图
  graph = builder.compile()
  
  # 执行图 - 小于等于5的情况
  print("输入值小于等于5:")
  result = graph.invoke({"value": 3})
  print(f"执行结果: {result}")
  
  # 执行图 - 大于5的情况
  print("\n输入值大于5:")
  result = graph.invoke({"value": 10})
  print(f"执行结果: {result}\n")

if __name__ == "__main__":
  main()




# === 条件入口点演示 ===
# 输入值小于等于5:
# 执行节点A
# 执行节点B
# 执行结果: {'value': 8, 'step': 'B执行完毕'}

# 输入值大于5:
# 执行节点D
# 执行节点B
# 执行结果: {'value': 40, 'step': 'B执行完毕'}