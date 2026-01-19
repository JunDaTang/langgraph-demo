"""
LangGraph 高级表单验证演示

该演示展示了如何验证来自人类的多个输入字段，
如果输入无效，就再次询问。通过在循环中多次调用 interrupt 来实现这一点。
"""

import sqlite3
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

class AdvancedFormState(TypedDict):
  """高级表单状态定义"""
  name: str | None
  age: int | None
  email: str | None

def get_name_node(state: AdvancedFormState):
  """
  获取姓名节点函数
  
  Args:
    state: 当前状态
    
  Returns:
    dict: 更新后的状态
  """
  print("执行节点: get_name_node")
  
  prompt = "请输入您的姓名:"

  while True:
    answer = interrupt(prompt)
    print(f"收到用户输入: {answer}")

    # 验证输入是否为非空字符串
    if isinstance(answer, str) and len(answer.strip()) > 0:
      name = answer.strip()
      print(f"姓名验证通过: {name}")
      return {"name": name}

    prompt = "姓名不能为空，请重新输入您的姓名:"

def get_age_node(state: AdvancedFormState):
  """
  获取年龄节点函数
  
  Args:
    state: 当前状态
    
  Returns:
    dict: 更新后的状态
  """
  print("执行节点: get_age_node")
  
  prompt = f"您好 {state['name']}，请输入您的年龄:"

  while True:
    answer = interrupt(prompt)
    print(f"收到用户输入: {answer}")

    # 验证输入是否为正整数
    if isinstance(answer, int) and answer > 0:
      print(f"年龄验证通过: {answer}")
      return {"age": answer}
    
    # 尝试将字符串转换为整数
    if isinstance(answer, str):
      try:
        age_int = int(answer)
        if age_int > 0:
          print(f"年龄验证通过: {age_int}")
          return {"age": age_int}
      except ValueError:
        pass

    prompt = f"'{answer}' 不是有效的年龄。请输入一个正整数:"

def get_email_node(state: AdvancedFormState):
  """
  获取邮箱节点函数
  
  Args:
    state: 当前状态
    
  Returns:
    dict: 更新后的状态
  """
  print("执行节点: get_email_node")
  
  prompt = f"您好 {state['name']} ({state['age']}岁)，请输入您的邮箱地址:"

  while True:
    answer = interrupt(prompt)
    print(f"收到用户输入: {answer}")

    # 简单验证邮箱格式
    if isinstance(answer, str) and "@" in answer and "." in answer:
      email = answer.strip()
      print(f"邮箱验证通过: {email}")
      return {"email": email}

    prompt = f"'{answer}' 不是有效的邮箱地址。请输入正确的邮箱格式 (example@domain.com):"

def main():
  """主函数 - 演示高级表单验证功能"""
  print("=== LangGraph 高级表单验证演示 ===\n")
  
  # 创建状态图
  builder = StateGraph(AdvancedFormState)
  builder.add_node("collect_name", get_name_node)
  builder.add_node("collect_age", get_age_node)
  builder.add_node("collect_email", get_email_node)
  
  builder.add_edge(START, "collect_name")
  builder.add_edge("collect_name", "collect_age")
  builder.add_edge("collect_age", "collect_email")
  builder.add_edge("collect_email", END)

  # 使用内存保存器作为检查点
  checkpointer = MemorySaver()
  
  # 编译图
  graph = builder.compile(checkpointer=checkpointer)

  # 配置线程ID
  config = {"configurable": {"thread_id": "advanced-form-1"}}
  
  # 初始化状态并执行图
  print("1. 启动高级表单收集工作流...")
  try:
    # 收集姓名
    state_after_name = graph.invoke({"name": None, "age": None, "email": None}, config=config)
    print(f"工作流中断信息: {state_after_name['__interrupt__']}\n")
    
    # 输入有效姓名
    print("2. 输入有效姓名...")
    state_after_name = graph.invoke(Command(resume="张三"), config=config)
    print(f"工作流中断信息: {state_after_name['__interrupt__']}\n")
    
    # 输入无效年龄
    print("3. 输入无效年龄...")
    state_after_invalid_age = graph.invoke(Command(resume="二十"), config=config)
    print(f"工作流中断信息: {state_after_invalid_age['__interrupt__']}\n")
    
    # 输入有效年龄
    print("4. 输入有效年龄...")
    state_after_age = graph.invoke(Command(resume=25), config=config)
    print(f"工作流中断信息: {state_after_age['__interrupt__']}\n")
    
    # 输入无效邮箱
    print("5. 输入无效邮箱...")
    state_after_invalid_email = graph.invoke(Command(resume="zhangsan"), config=config)
    print(f"工作流中断信息: {state_after_invalid_email['__interrupt__']}\n")
    
    # 输入有效邮箱
    print("6. 输入有效邮箱...")
    final_state = graph.invoke(Command(resume="zhangsan@example.com"), config=config)
    print(f"最终状态: {final_state}")
    print(f"姓名: {final_state['name']}")
    print(f"年龄: {final_state['age']}")
    print(f"邮箱: {final_state['email']}")
    print("\n=== 演示完成 ===")
    
  except Exception as e:
    print(f"执行过程中出现错误: {e}")
    print("\n=== 演示结束 ===")

if __name__ == "__main__":
  main()


# === LangGraph 高级表单验证演示 ===

# 1. 启动高级表单收集工作流...
# 执行节点: get_name_node
# 工作流中断信息: [Interrupt(value='请输入您的姓名:', id='82e8eb2c0dee3430d93d88f104dfe04f')]

# 2. 输入有效姓名...
# 执行节点: get_name_node
# 收到用户输入: 张三
# 姓名验证通过: 张三
# 执行节点: get_age_node
# 工作流中断信息: [Interrupt(value='您好 张三，请输入您的年龄:', id='53357d9485415bba6eaadb16b00dbfc7')]

# 3. 输入无效年龄...
# 执行节点: get_age_node
# 收到用户输入: 二十
# 工作流中断信息: [Interrupt(value="'二十' 不是有效的年龄。请输入一个正整数:", id='53357d9485415bba6eaadb16b00dbfc7')]

# 4. 输入有效年龄...
# 执行节点: get_age_node
# 收到用户输入: 二十
# 收到用户输入: 25
# 年龄验证通过: 25
# 执行节点: get_email_node
# 工作流中断信息: [Interrupt(value='您好 张三 (25岁)，请输入您的邮箱地址:', id='bb3536bf44613e0cbae2d168d4d4a79a')]

# 5. 输入无效邮箱...
# 执行节点: get_email_node
# 收到用户输入: zhangsan
# 工作流中断信息: [Interrupt(value="'zhangsan' 不是有效的邮箱地址。请输入正确的邮箱格式 (example@domain.com):", id='bb3536bf44613e0cbae2d168d4d4a79a')]

# 6. 输入有效邮箱...
# 执行节点: get_email_node
# 收到用户输入: zhangsan
# 收到用户输入: zhangsan@example.com
# 邮箱验证通过: zhangsan@example.com
# 最终状态: {'name': '张三', 'age': 25, 'email': 'zhangsan@example.com'}
# 姓名: 张三
# 年龄: 25
# 邮箱: zhangsan@example.com

# === 演示完成 ===