"""
LangGraph Reducer函数演示 - operator.mul Reducer（数值相乘）
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 4. operator.mul Reducer（数值相乘）
class MultiplyState(TypedDict):
  factor: Annotated[float, operator.mul]

def multiplier(state: MultiplyState) -> dict:
  return {"factor": 2.0}

def run_demo():
  """
  ！！！operator.mul实际使用上，官方设计存在bug：！！！
  在执行初始阶段（我们定义的第一个node前），会默认调用一次reducer（后面自定义reducer案例中进行了打印验证），用默认值与invoke传递的值进行计算：
  此案例中，invoke中传递了一个默认值5.0，由于会默认调用一次reducer，执行的计算是： 0.0（float默认值） * 5.0(invoke传递的初始值) = 0.0
  导致后续乘法结果一直都是0

  解决方案： 使用自定义reducer
  """
  print("4. operator.mul Reducer（数值相乘）演示:")
  builder = StateGraph(MultiplyState)
  builder.add_node("multiplier", multiplier)
  builder.add_edge(START, "multiplier")
  builder.add_edge("multiplier", END)
  graph = builder.compile()

  result = graph.invoke({"factor": 5.0})
  print(f"初始状态: {{'factor': 5.0}}")
  print(f"执行结果: {result}\n")

if __name__ == "__main__":
  run_demo()