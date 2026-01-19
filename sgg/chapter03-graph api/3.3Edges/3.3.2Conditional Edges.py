"""
LangGraph条件边演示

条件边根据当前状态动态决定下一个要执行的节点。
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

def node_c(state: GraphState) -> dict:
  """节点C"""
  print("执行节点C")
  return {"value": state["value"] - 1, "step": "C执行完毕"}

# 条件边的路由函数
def route_condition(state: GraphState) -> Literal["node_b", "node_c"]:
  """根据value值决定路由到哪个节点"""
  if state["value"] % 2 == 0:
    return "node_b_alias" # 偶数路由到节点B
  else:
    return "node_c_alias" # 奇数路由到节点C

def main():
  """演示条件边"""
  print("=== 条件边演示 ===")
  
  # 创建图
  builder = StateGraph(GraphState)
  
  # 添加节点
  builder.add_node("node_a", node_a)
  builder.add_node("node_b", node_b)
  builder.add_node("node_c", node_c)
  
  # 添加边
  builder.add_edge(START, "node_a") # 入口点
  
  # 添加条件边
  builder.add_conditional_edges(
    "node_a", # 源节点
    route_condition, # 路由函数
    { # 路由映射
      "node_b_alias": "node_b",
      "node_c_alias": "node_c"
    }
  )
  
  # 从B和C到结束
  builder.add_edge("node_b", END)
  builder.add_edge("node_c", END)
  
  # 编译图
  graph = builder.compile()
  
  # 执行图 - 偶数情况
  print("输入值为偶数:")
  result = graph.invoke({"value": 2})
  print(f"执行结果: {result}")
  
  # 执行图 - 奇数情况
  print("\n输入值为奇数:")
  result = graph.invoke({"value": 1})
  print(f"执行结果: {result}\n")

if __name__ == "__main__":
  main()



#   === 条件边演示 ===
# 输入值为偶数:
# 执行节点A
# 执行节点C
# 执行结果: {'value': 2, 'step': 'C执行完毕'}

# 输入值为奇数:
# 执行节点A
# 执行节点B
# 执行结果: {'value': 4, 'step': 'B执行完毕'}
