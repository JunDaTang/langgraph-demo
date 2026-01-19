#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 子图流式传输演示
展示如何在流式输出中包含子图的输出
"""

from langgraph.graph import START, StateGraph, END
from typing import TypedDict

# 定义子图状态
class SubgraphState(TypedDict):
  foo: str # 注意这个键与父图状态共享
  bar: str

def subgraph_node_1(state: SubgraphState):
  """子图节点1"""
  print(f" 执行子图节点1，当前状态: {state}")
  return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
  """子图节点2"""
  print(f" 执行子图节点2，当前状态: {state}")
  return {"foo": state["foo"] + state["bar"]}

# 定义父图状态
class ParentState(TypedDict):
  foo: str

def node_1(state: ParentState):
  """父图节点1"""
  print(f" 执行父图节点1，当前状态: {state}")
  return {"foo": "hi! " + state["foo"]}

def main():
  print("=== LangGraph 子图流式传输演示 ===\n")
  
  # 创建子图
  subgraph_builder = StateGraph(SubgraphState)
  subgraph_builder.add_node("subgraph_node_1", subgraph_node_1)
  subgraph_builder.add_node("subgraph_node_2", subgraph_node_2)
  subgraph_builder.add_edge(START, "subgraph_node_1")
  subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
  subgraph_builder.add_edge("subgraph_node_2", END)
  subgraph = subgraph_builder.compile()
  
  # 创建父图
  builder = StateGraph(ParentState)
  builder.add_node("node_1", node_1)
  builder.add_node("node_2", subgraph) # 将子图作为节点添加到父图中
  builder.add_edge(START, "node_1")
  builder.add_edge("node_1", "node_2")
  graph = builder.compile()
  
  print("--- 1. 不包含子图的常规流式输出 ---")
  for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="updates"
  ):
    print(f" 流式输出块: {chunk}")
  
  print("\n" + "="*50 + "\n")
  
  print("--- 2. 包含子图的流式输出 (subgraphs=True) ---")
  for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    # 设置 subgraphs=True 来流式传输子图的输出
    subgraphs=True, 
  ):
    print(f" 流式输出块: {chunk}")
  
  print("\n" + "="*50 + "\n")
  
  print("--- 3. 使用 values 模式并包含子图输出 ---")
  for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="values",
    subgraphs=True
  ):
    print(f" 流式输出块: {chunk}")
  
  print("\n" + "="*50 + "\n")
  
  print("--- 4. 详细分析子图流式输出 ---")
  print("当 subgraphs=True 时，输出格式为 (namespace, chunk) 元组:")
  for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    subgraphs=True
  ):
    namespace, data = chunk
    if namespace:
      print(f" 子图 '{namespace[0]}' 输出: {data}")
    else:
      print(f" 父图输出: {data}")

if __name__ == "__main__":
  main()




# === LangGraph 子图流式传输演示 ===

# --- 1. 不包含子图的常规流式输出 ---
#  执行父图节点1，当前状态: {'foo': 'foo'}
#  流式输出块: {'node_1': {'foo': 'hi! foo'}}
#  执行子图节点1，当前状态: {'foo': 'hi! foo'}
#  执行子图节点2，当前状态: {'foo': 'hi! foo', 'bar': 'bar'}
#  流式输出块: {'node_2': {'foo': 'hi! foobar'}}

# ==================================================

# --- 2. 包含子图的流式输出 (subgraphs=True) ---
#  执行父图节点1，当前状态: {'foo': 'foo'}
#  流式输出块: ((), {'node_1': {'foo': 'hi! foo'}})
#  执行子图节点1，当前状态: {'foo': 'hi! foo'}
#  执行子图节点2，当前状态: {'foo': 'hi! foo', 'bar': 'bar'}
#  流式输出块: (('node_2:e8044051-9f5b-bcd3-d702-cc32e81b4150',), {'subgraph_node_1': {'bar': 'bar'}})
#  流式输出块: (('node_2:e8044051-9f5b-bcd3-d702-cc32e81b4150',), {'subgraph_node_2': {'foo': 'hi! foobar'}})
#  流式输出块: ((), {'node_2': {'foo': 'hi! foobar'}})

# ==================================================

# --- 3. 使用 values 模式并包含子图输出 ---
#  流式输出块: ((), {'foo': 'foo'})
#  执行父图节点1，当前状态: {'foo': 'foo'}
#  流式输出块: ((), {'foo': 'hi! foo'})
#  执行子图节点1，当前状态: {'foo': 'hi! foo'}
#  流式输出块: (('node_2:6f83c15b-ab45-891b-5762-1f976a519264',), {'foo': 'hi! foo'})
#  执行子图节点2，当前状态: {'foo': 'hi! foo', 'bar': 'bar'}
#  流式输出块: (('node_2:6f83c15b-ab45-891b-5762-1f976a519264',), {'foo': 'hi! foo', 'bar': 'bar'})
#  流式输出块: (('node_2:6f83c15b-ab45-891b-5762-1f976a519264',), {'foo': 'hi! foobar', 'bar': 'bar'})
#  流式输出块: ((), {'foo': 'hi! foobar'})

# ==================================================

# --- 4. 详细分析子图流式输出 ---
# 当 subgraphs=True 时，输出格式为 (namespace, chunk) 元组:
#  执行父图节点1，当前状态: {'foo': 'foo'}
#  父图输出: {'node_1': {'foo': 'hi! foo'}}
#  执行子图节点1，当前状态: {'foo': 'hi! foo'}
#  执行子图节点2，当前状态: {'foo': 'hi! foo', 'bar': 'bar'}
#  子图 'node_2:60790c8a-569e-ba8b-89f5-058dd5971740' 输出: {'subgraph_node_1': {'bar': 'bar'}}
#  子图 'node_2:60790c8a-569e-ba8b-89f5-058dd5971740' 输出: {'subgraph_node_2': {'foo': 'hi! foobar'}}
#  父图输出: {'node_2': {'foo': 'hi! foobar'}}