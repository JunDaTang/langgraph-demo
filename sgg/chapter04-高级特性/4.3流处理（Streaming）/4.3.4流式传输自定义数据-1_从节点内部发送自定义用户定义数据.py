#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 自定义数据流式传输演示
展示如何从节点内部发送自定义用户定义数据
"""

from typing import TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
  query: str
  answer: str
  progress: list

def node_with_custom_streaming(state: State) -> State:
  """带自定义流式传输的节点"""
  # 获取流写入器以发送自定义数据
  writer = get_stream_writer()
  
  # 发送自定义数据（例如，进度更新）
  writer({"custom_key": "开始处理查询"})
  writer({"progress": "步骤1: 分析查询内容", "status": "running"})
  
  query = state["query"]
  # 模拟处理过程
  result = f"处理结果: {query.upper()}"
  
  writer({"progress": "步骤2: 生成结果", "status": "running"})
  writer({"progress": "步骤3: 完成处理", "status": "completed"})
  writer({"custom_key": "查询处理完成"})
  
  return {
    "answer": result,
    "progress": state.get("progress", []) + ["处理完成"]
  }

def main():
  print("=== LangGraph 自定义数据流式传输演示 ===\n")
  
  # 构建图
  graph = (
    StateGraph(State)
    .add_node("node_with_custom_streaming", node_with_custom_streaming)
    .add_edge(START, "node_with_custom_streaming")
    .add_edge("node_with_custom_streaming", END)
    .compile()
  )
  
  inputs = {"query": "hello world", "answer": "", "progress": []}
  
  print("--- 1. 使用 custom 流模式 ---")
  try:
    # 设置 stream_mode="custom" 以在流中接收自定义数据
    for chunk in graph.stream(inputs, stream_mode="custom"):
      print(f"自定义数据块: {chunk}")
  except Exception as e:
    print(f"错误: {e}")
    print("说明: 在Graph API中，自定义流数据需要在节点中通过特定方式发送")
  
  print("\n" + "="*50 + "\n")
  
  print("--- 2. 使用 updates 流模式 ---")
  for chunk in graph.stream(inputs, stream_mode="updates"):
    print(f"状态更新: {chunk}")
  
  print("\n" + "="*50 + "\n")
  
  print("--- 3. 同时使用 custom 和 updates 流模式 ---")
  try:
    for mode, chunk in graph.stream(inputs, stream_mode=["custom", "updates"]):
      print(f"[{mode}]: {chunk}")
  except Exception as e:
    print(f"错误: {e}")
    print("说明: 在Graph API中，需要特殊配置才能使用自定义流模式")

if __name__ == "__main__":
  main()



# === LangGraph 自定义数据流式传输演示 ===

# --- 1. 使用 custom 流模式 ---
# 自定义数据块: {'custom_key': '开始处理查询'}
# 自定义数据块: {'progress': '步骤1: 分析查询内容', 'status': 'running'}
# 自定义数据块: {'progress': '步骤2: 生成结果', 'status': 'running'}
# 自定义数据块: {'progress': '步骤3: 完成处理', 'status': 'completed'}
# 自定义数据块: {'custom_key': '查询处理完成'}

# ==================================================

# --- 2. 使用 updates 流模式 ---
# 状态更新: {'node_with_custom_streaming': {'answer': '处理结果: HELLO WORLD', 'progress': ['处理完成']}}

# ==================================================

# --- 3. 同时使用 custom 和 updates 流模式 ---
# [custom]: {'custom_key': '开始处理查询'}
# [custom]: {'progress': '步骤1: 分析查询内容', 'status': 'running'}
# [custom]: {'progress': '步骤2: 生成结果', 'status': 'running'}
# [custom]: {'progress': '步骤3: 完成处理', 'status': 'completed'}
# [custom]: {'custom_key': '查询处理完成'}
# [updates]: {'node_with_custom_streaming': {'answer': '处理结果: HELLO WORLD', 'progress': ['处理完成']}}