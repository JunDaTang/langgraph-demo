"""
LangGraph Reducer函数演示 - 数值累加Reducer
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 7. 数值累加Reducer
class NumberAddState(TypedDict):
  count: Annotated[int, operator.add]

def increment_1(state: NumberAddState) -> dict:
  return {"count": 5}

def increment_2(state: NumberAddState) -> dict:
  return {"count": 3}

def run_demo():
  print("3.3 数值累加Reducer演示:")
  builder = StateGraph(NumberAddState)
  builder.add_node("increment_1", increment_1)
  builder.add_node("increment_2", increment_2)
  builder.add_edge(START, "increment_1")
  builder.add_edge(START, "increment_2") # 并行执行
  builder.add_edge("increment_1", END)
  builder.add_edge("increment_2", END)
  graph = builder.compile()
  
  result = graph.invoke({"count": 10})
  print(f"初始状态: {{'count': 10}}")
  print(f"执行结果: {result}\n")

if __name__ == "__main__":
  run_demo()