"""
简单的 LangGraph 可视化演示

这个演示展示了一个更简单的图结构，用于更好地理解可视化功能
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 定义状态
class SimpleState(TypedDict):
  value: int
  messages: Annotated[list, lambda x, y: x + y]

# 定义节点函数
def node_a(state: SimpleState) -> dict:
  """节点A"""
  print("执行节点A")
  return {
    "value": state["value"] + 1,
    "messages": [("system", "执行了节点A")]
  }

def node_b(state: SimpleState) -> dict:
  """节点B"""
  print("执行节点B")
  return {
    "value": state["value"] * 2,
    "messages": [("system", "执行了节点B")]
  }

def node_c(state: SimpleState) -> dict:
  """节点C"""
  print("执行节点C")
  return {
    "value": state["value"] + 10,
    "messages": [("system", "执行了节点C")]
  }

def main():
  """简单可视化演示"""
  print("=== 简单 LangGraph 可视化演示 ===\n")
  
  # 创建图
  builder = StateGraph(SimpleState)
  
  # 添加节点
  builder.add_node("node_a", node_a)
  builder.add_node("node_b", node_b)
  builder.add_node("node_c", node_c)
  
  # 添加边
  builder.add_edge(START, "node_a")
  builder.add_edge("node_a", "node_b")
  builder.add_edge("node_b", "node_c")
  builder.add_edge("node_c", END)
  
  # 编译图
  graph = builder.compile()
  
  # 获取图结构
  graph_structure = graph.get_graph()
  
  print("1. Mermaid 图表代码:")
  try:
    mermaid_code = graph_structure.draw_mermaid()
    print(mermaid_code)
  except Exception as e:
    print(f"生成 Mermaid 图表失败: {e}")
  print()
  
  print("2. ASCII 文本可视化（需要安装 grandalf）:")
  try:
    ascii_graph = graph_structure.draw_ascii()
    print(ascii_graph)
  except Exception as e:
    print(f"ASCII 可视化失败: {e}")
    print("提示：可以通过以下命令安装 grandalf 来支持 ASCII 可视化:")
    print(" pip install grandalf")
  print()
  
  print("3. 执行图:")
  initial_state = {
    "value": 5,
    "messages": []
  }
  
  print("初始状态:", initial_state)
  result = graph.invoke(initial_state)
  print("最终状态:", result)

if __name__ == "__main__":
  main()



# === 简单 LangGraph 可视化演示 ===

# 1. Mermaid 图表代码:
# ---
# config:
#   flowchart:
#     curve: linear
# ---
# graph TD;
#         __start__([<p>__start__</p>]):::first
#         node_a(node_a)
#         node_b(node_b)
#         node_c(node_c)
#         __end__([<p>__end__</p>]):::last
#         __start__ --> node_a;
#         node_a --> node_b;
#         node_b --> node_c;
#         node_c --> __end__;
#         classDef default fill:#f2f0ff,line-height:1.2
#         classDef first fill-opacity:0
#         classDef last fill:#bfb6fc


# 2. ASCII 文本可视化（需要安装 grandalf）:
# +-----------+  
# | __start__ |
# +-----------+
#       *
#       *
#       *
#   +--------+
#   | node_a |
#   +--------+
#       *
#       *
#       *
#   +--------+
#   | node_b |
#   +--------+
#       *
#       *
#       *
#   +--------+
#   | node_c |
#   +--------+
#       *
#       *
#       *
#  +---------+
#  | __end__ |
#  +---------+

# 3. 执行图:
# 初始状态: {'value': 5, 'messages': []}
# 执行节点A
# 执行节点B
# 执行节点C
# 最终状态: {'value': 22, 'messages': [('system', '执行了节点A'), ('system', '执行了节点B'), ('system', '执行了节点C')]}