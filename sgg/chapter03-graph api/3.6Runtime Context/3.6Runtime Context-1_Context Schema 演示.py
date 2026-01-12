"""
LangGraph Context Schema 演示

演示如何在 LangGraph 1.0 中使用 context_schema 向节点传递不属于图表状态的信息。
这在传递模型名称、数据库连接等依赖项时非常有用。
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, HumanMessage
from dataclasses import dataclass

# 定义状态结构
class AgentState(TypedDict):
  messages: Annotated[list, lambda x, y: x + y]
  response: str

# 定义上下文结构
@dataclass
class ContextSchema:
  model_name: str
  db_connection: str
  api_key: str

# 节点函数：处理用户消息
def process_message(state: AgentState, runtime: Runtime[ContextSchema]) -> dict:
  """处理用户消息的节点，使用context中的信息"""
  print("执行节点: process_message")
  
  # 获取最新的用户消息
  last_message = state["messages"][-1].content if state["messages"] else ""
  print(f"用户消息: {last_message}")
  
  # 使用runtime.context中的信息
  model_name = runtime.context.model_name
  db_connection = runtime.context.db_connection
  api_key = runtime.context.api_key
  
  print(f"使用的模型: {model_name}")
  print(f"数据库连接: {db_connection}")
  print(f"API密钥前缀: {api_key[:5]}***") # 只显示前5位，隐藏其余部分
  
  # 模拟使用这些信息处理请求
  response = f"使用 {model_name} 处理了您的请求，已连接到 {db_connection}"
  
  return {
    "messages": [AIMessage(content=response)],
    "response": response
  }

# 节点函数：生成最终响应
def generate_response(state: AgentState, runtime: Runtime[ContextSchema]) -> dict:
  """生成最终响应的节点"""
  print("执行节点: generate_response")
  
  # 使用runtime.context中的信息
  model_name = runtime.context.model_name
  print(f"使用模型 {model_name} 生成最终响应")
  
  # 获取之前的结果
  previous_response = state["response"]
  
  # 生成更详细的响应
  final_response = f"{previous_response}\n\n这是使用 {model_name} 生成的完整响应。"
  
  return {
    "messages": [AIMessage(content=final_response)],
    "response": final_response
  }

def main():
  """演示 context_schema 的使用"""
  print("=== Context Schema 演示 ===\n")
  
  # 创建图，指定state_schema和context_schema
  builder = StateGraph(AgentState, context_schema=ContextSchema)
  
  # 添加节点
  builder.add_node("process_message", process_message)
  builder.add_node("generate_response", generate_response)
  
  # 添加边
  builder.add_edge(START, "process_message")
  builder.add_edge("process_message", "generate_response")
  builder.add_edge("generate_response", END)
  
  # 编译图
  graph = builder.compile()
  
  # 定义初始状态
  initial_state = {
    "messages": [HumanMessage(content="请帮我查询最新的订单信息")],
    "response": ""
  }
  
  # 定义上下文
  context = ContextSchema(
    model_name="gpt-4-turbo",
    db_connection="postgresql://user:pass@localhost:5432/orders_db",
    api_key="sk-abcdefghijklmnopqrstuvwxyz123456"
  )
  
  print("初始状态:", initial_state)
  print("上下文信息:", {
    "model_name": context.model_name,
    "db_connection": context.db_connection,
    "api_key": f"{context.api_key[:5]}***"
  })
  print("\n" + "-"*50 + "\n")
  
  # 执行图，通过context参数传递上下文
  result = graph.invoke(initial_state, context=context)
  
  print("\n" + "="*50)
  print("最终状态:", result)
  print("\n最终响应:")
  print(result["response"])

if __name__ == "__main__":
  main()