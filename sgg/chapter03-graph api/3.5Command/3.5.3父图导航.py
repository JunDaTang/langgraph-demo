"""
LangGraph Command 父图导航演示

演示如何使用 Command 对象从子图导航到父图节点。
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# 定义父图状态
class ParentState(TypedDict):
  messages: Annotated[list, lambda x, y: x + y]
  task_status: str
  subtask_result: str

# 定义子图状态（继承父图状态）
class ChildState(TypedDict):
  messages: Annotated[list, lambda x, y: x + y]
  task_status: str
  subtask_result: str
  child_data: str

# 父图节点：主控制器
def main_controller(state: ParentState) -> Command[ParentState]:
  """主控制器节点"""
  print("执行节点: main_controller (父图)")
  
  # 启动子任务
  return Command(
    update={
      "messages": [("system", "启动子任务")],
      "task_status": "subtask_started"
    },
    goto="subgraph_node"
  )

# 父图节点：任务结束
def task_finisher(state: ParentState) -> dict:
  """任务结束节点"""
  print("执行节点: task_finisher (父图)")
  
  return {
    "messages": [("system", "任务完成")],
    "task_status": "completed"
  }

# 子图节点：数据处理器
def data_processor(state: ChildState) -> Command[ParentState]:
  """数据处理器节点（在子图中）"""
  print("执行节点: data_processor (子图)")
  
  # 处理数据
  processed_data = "处理后的数据"
  print(f"处理结果: {processed_data}")
  
  # 导航回父图的task_finisher节点
  return Command(
    update={
      "messages": [("subtask", f"子任务完成: {processed_data}")],
      "subtask_result": processed_data,
      "task_status": "subtask_completed"
    },
    goto="task_finisher",
    graph=Command.PARENT # 指定导航到父图
  )

def create_subgraph() -> StateGraph:
  """创建子图"""
  subgraph_builder = StateGraph(ChildState)
  subgraph_builder.add_node("data_processor", data_processor)
  subgraph_builder.add_edge(START, "data_processor")
  subgraph_builder.add_edge("data_processor", END)
  return subgraph_builder.compile() # 编译子图

def main():
  """演示Command父图导航"""
  print("=== Command 父图导航演示 ===\n")
  
  # 创建父图
  parent_builder = StateGraph(ParentState)
  
  # 添加节点
  parent_builder.add_node("main_controller", main_controller)
  parent_builder.add_node("task_finisher", task_finisher)
  parent_builder.add_node("subgraph_node", create_subgraph()) # 添加子图作为节点
  
  # 添加边
  parent_builder.add_edge(START, "main_controller")
  parent_builder.add_edge("main_controller", "subgraph_node")
  
  # 编译图
  graph = parent_builder.compile()
  
  # 执行图
  initial_state = {
    "messages": [("user", "开始任务")],
    "task_status": "init",
    "subtask_result": ""
  }
  print("初始状态:", initial_state)
  
  result = graph.invoke(initial_state)
  print("最终状态:", result)

if __name__ == "__main__":
  main()