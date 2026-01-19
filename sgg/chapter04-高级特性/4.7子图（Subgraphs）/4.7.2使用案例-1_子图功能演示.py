"""
LangGraph 子图功能演示

该演示展示了如何在 LangGraph 中使用子图，包括：
1. 从节点调用图（不同的状态模式）
2. 将图添加为节点（共享状态模式）
3. 查看子图状态
4. 流式输出子图结果
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

# 定义子图状态（不同的状态模式）
class SubgraphState(TypedDict):
  bar: str
  baz: str

# 定义父图状态（不同的状态模式）
class ParentState(TypedDict):
  foo: str

# 定义共享状态的子图
class SharedSubgraphState(TypedDict):
  foo: str # 共享状态键
  bar: str # 子图私有状态键

# 定义用于中断演示的状态
class InterruptState(TypedDict):
  foo: str

def subgraph_node_1(state: SubgraphState):
  """子图节点1"""
  print("执行子图节点1")
  return {"baz": "baz"}

def subgraph_node_2(state: SubgraphState):
  """子图节点2"""
  print("执行子图节点2")
  return {"bar": state["bar"] + state["baz"]}

def shared_subgraph_node_1(state: SharedSubgraphState):
  """共享状态子图节点1"""
  print("执行共享状态子图节点1")
  return {"bar": "bar"}

def shared_subgraph_node_2(state: SharedSubgraphState):
  """共享状态子图节点2"""
  print("执行共享状态子图节点2")
  return {"foo": state["foo"] + state["bar"]}

def interrupt_subgraph_node(state: InterruptState):
  """用于中断演示的子图节点"""
  print("执行中断子图节点")
  # 模拟中断，实际应用中会使用 interrupt() 函数
  user_input = input("请输入值（模拟中断）: ")
  return {"foo": state["foo"] + user_input}

def create_subgraph_different_schemas():
  """创建具有不同状态模式的子图"""
  print("\\n=== 创建具有不同状态模式的子图 ===")
  subgraph_builder = StateGraph(SubgraphState)
  subgraph_builder.add_node("subgraph_node_1", subgraph_node_1)
  subgraph_builder.add_node("subgraph_node_2", subgraph_node_2)
  subgraph_builder.add_edge(START, "subgraph_node_1")
  subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
  return subgraph_builder.compile()

def node_1(state: ParentState):
  """父图节点1"""
  print("执行父图节点1")
  return {"foo": "hi! " + state["foo"]}

def node_2(subgraph):
  """父图节点2 - 调用子图"""
  def _call_subgraph(state: ParentState):
    print("执行父图节点2（调用子图）")
    # 转换状态到子图格式
    subgraph_input = {"bar": state["foo"], "baz": ""}
    response = subgraph.invoke(subgraph_input)
    # 转换响应回父图格式
    return {"foo": response["bar"]}
  return _call_subgraph

def create_parent_graph_with_subgraph_call(subgraph):
  """创建通过节点调用子图的父图"""
  print("\\n=== 创建通过节点调用子图的父图 ===")
  builder = StateGraph(ParentState)
  builder.add_node("node_1", node_1)
  builder.add_node("node_2", node_2(subgraph))
  builder.add_edge(START, "node_1")
  builder.add_edge("node_1", "node_2")
  return builder.compile()

def create_shared_subgraph():
  """创建具有共享状态的子图"""
  print("\\n=== 创建具有共享状态的子图 ===")
  subgraph_builder = StateGraph(SharedSubgraphState)
  subgraph_builder.add_node("shared_subgraph_node_1", shared_subgraph_node_1)
  subgraph_builder.add_node("shared_subgraph_node_2", shared_subgraph_node_2)
  subgraph_builder.add_edge(START, "shared_subgraph_node_1")
  subgraph_builder.add_edge("shared_subgraph_node_1", "shared_subgraph_node_2")
  return subgraph_builder.compile()

def create_parent_graph_with_node_subgraph(subgraph):
  """创建将子图作为节点添加的父图"""
  print("\\n=== 创建将子图作为节点添加的父图 ===")
  builder = StateGraph(ParentState)
  builder.add_node("node_1", node_1)
  builder.add_node("node_2", subgraph) # 直接将子图作为节点添加
  builder.add_edge(START, "node_1")
  builder.add_edge("node_1", "node_2")
  return builder.compile()

def create_interrupt_subgraph():
  """创建用于中断演示的子图"""
  print("\\n=== 创建用于中断演示的子图 ===")
  subgraph_builder = StateGraph(InterruptState)
  subgraph_builder.add_node("interrupt_subgraph_node", interrupt_subgraph_node)
  subgraph_builder.add_edge(START, "interrupt_subgraph_node")
  return subgraph_builder.compile()

def create_parent_graph_with_interrupt_subgraph(subgraph):
  """创建包含中断子图的父图"""
  print("\\n=== 创建包含中断子图的父图 ===")
  builder = StateGraph(InterruptState)
  builder.add_node("node_1", subgraph)
  builder.add_edge(START, "node_1")
  return builder.compile()

def demo_subgraph_call():
  """演示从节点调用图"""
  print("\\n=== 演示从节点调用图 ===")
  subgraph = create_subgraph_different_schemas()
  parent_graph = create_parent_graph_with_subgraph_call(subgraph)
  
  print("开始执行图:")
  for chunk in parent_graph.stream({"foo": "foo"}, subgraphs=True):
    print(f"流式输出: {chunk}")

def demo_add_graph_as_node():
  """演示将图添加为节点"""
  print("\\n=== 演示将图添加为节点 ===")
  subgraph = create_shared_subgraph()
  parent_graph = create_parent_graph_with_node_subgraph(subgraph)
  
  print("开始执行图:")
  for chunk in parent_graph.stream({"foo": "foo"}):
    print(f"流式输出: {chunk}")

def demo_subgraph_streaming():
  """演示流式输出子图结果"""
  print("\\n=== 演示流式输出子图结果 ===")
  subgraph = create_shared_subgraph()
  parent_graph = create_parent_graph_with_node_subgraph(subgraph)
  
  print("开始流式执行图:")
  for chunk in parent_graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    subgraphs=True, 
  ):
    print(f"流式输出: {chunk}")

def main():
  """主函数"""
  print("=== LangGraph 子图功能演示 ===")
  
  # 演示从节点调用图
  demo_subgraph_call()
  
  print("\\n" + "="*50 + "\\n")
  
  # 演示将图添加为节点
  demo_add_graph_as_node()
  
  print("\\n" + "="*50 + "\\n")
  
  # 演示流式输出子图结果
  demo_subgraph_streaming()
  
  print("\\n=== 演示完成 ===")

if __name__ == "__main__":
  main()

### 问题：stream({"foo": "foo"}, subgraphs=True)默认是stream_mode="updates"？

# === LangGraph 子图功能演示 ===
# \n=== 演示从节点调用图 ===
# \n=== 创建具有不同状态模式的子图 ===
# \n=== 创建通过节点调用子图的父图 ===
# 开始执行图:
# 执行父图节点1
# 流式输出: ((), {'node_1': {'foo': 'hi! foo'}})
# 执行父图节点2（调用子图）
# 执行子图节点1
# 执行子图节点2
# 流式输出: (('node_2:e17e024f-2544-6ebd-0572-2c7665e0a5ba',), {'subgraph_node_1': {'baz': 'baz'}})
# 流式输出: (('node_2:e17e024f-2544-6ebd-0572-2c7665e0a5ba',), {'subgraph_node_2': {'bar': 'hi! foobaz'}})
# 流式输出: ((), {'node_2': {'foo': 'hi! foobaz'}})
# \n==================================================\n
# \n=== 演示将图添加为节点 ===
# \n=== 创建具有共享状态的子图 ===
# \n=== 创建将子图作为节点添加的父图 ===
# 开始执行图:
# 执行父图节点1
# 流式输出: {'node_1': {'foo': 'hi! foo'}}
# 执行共享状态子图节点1
# 执行共享状态子图节点2
# 流式输出: {'node_2': {'foo': 'hi! foobar'}}
# \n==================================================\n
# \n=== 演示流式输出子图结果 ===
# \n=== 创建具有共享状态的子图 ===
# \n=== 创建将子图作为节点添加的父图 ===
# 开始流式执行图:
# 执行父图节点1
# 流式输出: ((), {'node_1': {'foo': 'hi! foo'}})
# 执行共享状态子图节点1
# 执行共享状态子图节点2
# 流式输出: (('node_2:c029a8ee-1adb-7283-03e3-9e30920ab70d',), {'shared_subgraph_node_1': {'bar': 'bar'}})
# 流式输出: (('node_2:c029a8ee-1adb-7283-03e3-9e30920ab70d',), {'shared_subgraph_node_2': {'foo': 'hi! foobar'}})
# 流式输出: ((), {'node_2': {'foo': 'hi! foobar'}})
# \n=== 演示完成 ===
