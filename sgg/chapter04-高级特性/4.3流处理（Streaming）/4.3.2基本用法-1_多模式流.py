#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph 多模式流式传输演示
展示LangGraph支持的各种流模式
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 定义状态类型
class State(TypedDict):
  question: str
  answer: str
  confidence: float
  steps: list

def think(state: State) -> State:
  """思考节点"""
  question = state["question"]
  # 模拟思考过程
  steps = [f"分析问题: {question}", "检索相关知识", "形成初步答案"]
  return {"steps": steps}

def respond(state: State) -> State:
  """回应节点"""
  question = state["question"]
  # 根据问题生成答案
  if "天气" in question:
    answer = "今天天气晴朗"
    confidence = 0.9
  elif "时间" in question:
    answer = "现在是上午10点"
    confidence = 0.8
  else:
    answer = "这是一个很好的问题"
    confidence = 0.7
    
  return {
    "answer": answer,
    "confidence": confidence
  }

def reflect(state: State) -> State:
  """反思节点"""
  answer = state["answer"]
  confidence = state["confidence"]
  steps = state.get("steps", [])
  
  steps.append(f"验证答案: {answer}")
  steps.append(f"置信度评估: {confidence}")
  
  if confidence > 0.8:
    conclusion = "高置信度答案"
  elif confidence > 0.5:
    conclusion = "中等置信度答案"
  else:
    conclusion = "低置信度答案"
    
  steps.append(f"结论: {conclusion}")
  
  return {"steps": steps}

def main():
  # 构建图
  builder = StateGraph(State)
  builder.add_node("think", think)
  builder.add_node("respond", respond)
  builder.add_node("reflect", reflect)
  
  builder.add_edge(START, "think")
  builder.add_edge("think", "respond")
  builder.add_edge("respond", "reflect")
  builder.add_edge("reflect", END)
  
  graph = builder.compile()
  
  print("=== LangGraph 多模式流式传输演示 ===\n")
  
  # 准备输入
  input_state = {
    "question": "今天天气怎么样?",
    "answer": "",
    "confidence": 0.0,
    "steps": []
  }
  
  print("--- 1. 使用 stream_mode='values' 模式 ---")
  print("显示每一步执行后的完整状态:")
  for chunk in graph.stream(input_state, stream_mode="values"):
    print(f" {chunk}")
  
  print("\n" + "="*60 + "\n")
  
  print("--- 2. 使用 stream_mode='updates' 模式 ---")
  print("只显示每一步的状态更新:")
  for chunk in graph.stream(input_state, stream_mode="updates"):
    print(f" {chunk}")
  
  print("\n" + "="*60 + "\n")
  
  print("--- 3. 同时使用多种流模式 ---")
  print("同时显示完整状态和状态更新:")
  for mode, chunk in graph.stream(input_state, stream_mode=["values", "updates"]):
    print(f" [{mode}]: {chunk}")
  
  print("\n" + "="*60 + "\n")
  
  print("--- 4. 使用 debug 模式 ---")
  print("显示详细的调试信息:")
  try:
    for chunk in graph.stream(input_state, stream_mode="debug"):
      print(f" {chunk}")
  except Exception as e:
    print(f" Debug模式可能需要特殊配置: {e}")

if __name__ == "__main__":
  main()



# === LangGraph 多模式流式传输演示 ===

# --- 1. 使用 stream_mode='values' 模式 ---
# 显示每一步执行后的完整状态:
#  {'question': '今天天气怎么样?', 'answer': '', 'confidence': 0.0, 'steps': []}
#  {'question': '今天天气怎么样?', 'answer': '', 'confidence': 0.0, 'steps': ['分析问题: 今天天气怎么样?', '检索相关知识', '形成初
# 步答案']}
#  {'question': '今天天气怎么样?', 'answer': '今天天气晴朗', 'confidence': 0.9, 'steps': ['分析问题: 今天天气怎么样?', '检索相关知
# 识', '形成初步答案']}
#  {'question': '今天天气怎么样?', 'answer': '今天天气晴朗', 'confidence': 0.9, 'steps': ['分析问题: 今天天气怎么样?', '检索相关知
# 识', '形成初步答案', '验证答案: 今天天气晴朗', '置信度评估: 0.9', '结论: 高置信度答案']}

# ============================================================

# --- 2. 使用 stream_mode='updates' 模式 ---
# 只显示每一步的状态更新:
#  {'think': {'steps': ['分析问题: 今天天气怎么样?', '检索相关知识', '形成初步答案']}}
#  {'respond': {'answer': '今天天气晴朗', 'confidence': 0.9}}
#  {'reflect': {'steps': ['分析问题: 今天天气怎么样?', '检索相关知识', '形成初步答案', '验证答案: 今天天气晴朗', '置信度评估: 0.9', '结论: 高置信度答案']}}

# ============================================================

# --- 3. 同时使用多种流模式 ---
# 同时显示完整状态和状态更新:
#  [values]: {'question': '今天天气怎么样?', 'answer': '', 'confidence': 0.0, 'steps': []}
#  [updates]: {'think': {'steps': ['分析问题: 今天天气怎么样?', '检索相关知识', '形成初步答案']}}
#  [values]: {'question': '今天天气怎么样?', 'answer': '', 'confidence': 0.0, 'steps': ['分析问题: 今天天气怎么样?', '检索相关知识
# ', '形成初步答案']}
#  [updates]: {'respond': {'answer': '今天天气晴朗', 'confidence': 0.9}}
#  [values]: {'question': '今天天气怎么样?', 'answer': '今天天气晴朗', 'confidence': 0.9, 'steps': ['分析问题: 今天天气怎么样?', '检索相关知识', '形成初步答案']}
#  [updates]: {'reflect': {'steps': ['分析问题: 今天天气怎么样?', '检索相关知识', '形成初步答案', '验证答案: 今天天气晴朗', '置信 
# 度评估: 0.9', '结论: 高置信度答案']}}
#  [values]: {'question': '今天天气怎么样?', 'answer': '今天天气晴朗', 'confidence': 0.9, 'steps': ['分析问题: 今天天气怎么样?', '检索相关知识', '形成初步答案', '验证答案: 今天天气晴朗', '置信度评估: 0.9', '结论: 高置信度答案']}

# ============================================================

# --- 4. 使用 debug 模式 ---
# 显示详细的调试信息:
#  {'step': 1, 'timestamp': '2026-01-15T02:57:06.268091+00:00', 'type': 'task', 'payload': {'id': 'c3512e47-3cd7-9e24-88ee-75cd64acb9e4', 'name': 'think', 'input': {'question': '今天天气怎么样?', 'answer': '', 'confidence': 0.0, 'steps': []}, 'triggers': ('branch:to:think',)}}
#  {'step': 1, 'timestamp': '2026-01-15T02:57:06.268091+00:00', 'type': 'task_result', 'payload': {'id': 'c3512e47-3cd7-9e24-88ee-75cd64acb9e4', 'name': 'think', 'error': None, 'result': {'steps': ['分析问题: 今天天气怎么样?', '检索相关知识', '形成初步答案']}, 'interrupts': []}}
#  {'step': 2, 'timestamp': '2026-01-15T02:57:06.269091+00:00', 'type': 'task', 'payload': {'id': '5b296f88-8145-0246-5cf2-6943f494483c', 'name': 'respond', 'input': {'question': '今天天气怎么样?', 'answer': '', 'confidence': 0.0, 'steps': ['分析问题: 今天天
# 气怎么样?', '检索相关知识', '形成初步答案']}, 'triggers': ('branch:to:respond',)}}
#  {'step': 2, 'timestamp': '2026-01-15T02:57:06.271091+00:00', 'type': 'task_result', 'payload': {'id': '5b296f88-8145-0246-5cf2-6943f494483c', 'name': 'respond', 'error': None, 'result': {'answer': '今天天气晴朗', 'confidence': 0.9}, 'interrupts': []}}    
#  {'step': 3, 'timestamp': '2026-01-15T02:57:06.271091+00:00', 'type': 'task', 'payload': {'id': '7b9eb385-fa27-03da-abbf-d572ff39459e', 'name': 'reflect', 'input': {'question': '今天天气怎么样?', 'answer': '今天天气晴朗', 'confidence': 0.9, 'steps': ['分析
# 问题: 今天天气怎么样?', '检索相关知识', '形成初步答案']}, 'triggers': ('branch:to:reflect',)}}
#  {'step': 3, 'timestamp': '2026-01-15T02:57:06.272091+00:00', 'type': 'task_result', 'payload': {'id': '7b9eb385-fa27-03da-abbf-d572ff39459e', 'name': 'reflect', 'error': None, 'result': {'steps': ['分析问题: 今天天气怎么样?', '检索相关知识', '形成初步答案
# ', '验证答案: 今天天气晴朗', '置信度评估: 0.9', '结论: 高置信度答案']}, 'interrupts': []}}