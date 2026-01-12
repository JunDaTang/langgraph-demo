"""
LangGraph普通边演示

普通边是直接连接两个节点的边，表示无条件地从一个节点跳转到另一个节点。
"""

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

def main():
  """演示普通边"""
  print("=== 普通边演示 ===")
  
  # 创建图
  builder = StateGraph(GraphState)
  
  # 添加节点
  builder.add_node("node_a", node_a)
  builder.add_node("node_b", node_b)
  builder.add_node("node_c", node_c)
  
  # 添加普通边
  builder.add_edge(START, "node_a") # 从开始到A
  builder.add_edge("node_a", "node_b") # 从A到B
  builder.add_edge("node_b", "node_c") # 从B到C
  builder.add_edge("node_c", END)    # 从C到结束
  
  # 编译图
  graph = builder.compile()
  
  # 执行图
  result = graph.invoke({"value": 1})
  print(f"执行结果: {result}\n")

if __name__ == "__main__":
  main()




# === 普通边演示 ===
# 执行节点A
# 执行节点B
# 执行节点C
# 执行结果: {'value': 3, 'step': 'C执行完毕'}