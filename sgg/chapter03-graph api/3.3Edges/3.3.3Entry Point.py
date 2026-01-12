"""
LangGraph入口点演示

入口点定义了图开始执行的第一个节点。
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

def main():
  """演示入口点"""
  print("=== 入口点演示 ===")
  
  # 创建图
  builder = StateGraph(GraphState)
  
  # 添加节点
  builder.add_node("node_a", node_a)
  builder.add_node("node_b", node_b)
  
  # 设置入口点和出口点
  builder.add_edge(START, "node_a")
  builder.add_edge("node_a", "node_b")
  builder.add_edge("node_b", END)
  
  # 编译图
  graph = builder.compile()
  
  # 执行图
  result = graph.invoke({"value": 0})
  print(f"执行结果: {result}\n")

if __name__ == "__main__":
  main()



#   === 入口点演示 ===
# 执行节点A
# 执行节点B
# 执行结果: {'value': 2, 'step': 'B执行完毕'}