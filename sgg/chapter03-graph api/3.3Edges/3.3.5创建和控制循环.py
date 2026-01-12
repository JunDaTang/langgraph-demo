from typing import Annotated, Dict, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

class LoopState(TypedDict):
  count: int
  result: str
  max_count: int

def node_a(state: LoopState) -> dict:
  """节点a：处理逻辑并更新计数"""
  print(f"执行节点a，当前计数: {state['count']}")
  return {
    'count': state['count'] + 1,
    'result': f"已处理{state['count']}次"
  }

def node_b(state: LoopState) -> dict:
  """节点b：辅助处理"""
  print(f"执行节点b，当前计数: {state['count']}")
  return {
    'result': f"已处理{state['count']}次 - 辅助处理"
  }

def route(state: LoopState) -> Literal["b", END]:
  """条件路由函数：决定是继续循环还是终止"""
  # 终止条件：当计数达到最大值时终止
  if state['count'] >= state['max_count']:
    print(f"满足终止条件，计数 {state['count']} >= {state['max_count']}，返回END")
    return END
  else:
    print(f"未满足终止条件，计数 {state['count']} < {state['max_count']}，返回b")
    return "b"

# 创建图
builder = StateGraph(LoopState)

# 添加节点
builder.add_node("a", node_a)
builder.add_node("b", node_b)

# 添加边
builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")

# 编译图
graph = builder.compile()

# 执行图
print("=== 开始执行工作流 ===")
try:
  result = graph.invoke(input={
    'count': 0,
    'result': '',
    'max_count': 3
  }, config={
    'recursion_limit': 6 # 设置递归限制
  })
  print("=== 执行结果 ===")
  print(result)
except GraphRecursionError as e:
  print(f"递归错误: {e}")




# === 开始执行工作流 ===
# 执行节点a，当前计数: 0
# 未满足终止条件，计数 1 < 3，返回b
# 执行节点b，当前计数: 1
# 执行节点a，当前计数: 1
# 未满足终止条件，计数 2 < 3，返回b
# 执行节点b，当前计数: 2
# 执行节点a，当前计数: 2
# 满足终止条件，计数 3 >= 3，返回END
# === 执行结果 ===
# {'count': 3, 'result': '已处理2次', 'max_count': 3}