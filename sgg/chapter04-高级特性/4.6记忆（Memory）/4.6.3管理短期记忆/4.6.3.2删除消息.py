"""
LangGraph 消息删除演示

该演示展示了如何使用 RemoveMessage 从图状态中删除消息。
当状态的 key 带有 add_messages 这个 reducer 时（例如 MessagesState），RemoveMessage 可以正常工作。
"""

from typing import Annotated, Sequence
from langchain_core.messages import (
  HumanMessage, 
  AIMessage, 
  RemoveMessage,
  BaseMessage
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict
import os
import dotenv
dotenv.load_dotenv()  # 默认加载 .env

# 定义状态类型
class CustomMessagesState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], "messages"]

# 初始化模型（使用模拟模型）
model = None
try:
  # 尝试初始化百炼平台的通义大模型
  api_key = os.getenv("DASHSCOPE_API_KEY")
  
  model = init_chat_model(
    "qwen-plus",
    model_provider="openai", # 使用openai提供者，但配置为百炼平台
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7
  )
  print("成功初始化百炼平台的通义大模型")
except Exception as e:
  print(f"初始化模型失败: {e}")
  print("将使用模拟响应模式")

def call_model(state: MessagesState):
  """
  调用模型的节点函数
  
  Args:
    state: 当前状态，包含消息历史
    
  Returns:
    dict: 更新后的状态
  """
  print("\\n执行节点: call_model")
  print(f"当前消息数量: {len(state['messages'])}")
  
  # 显示所有消息
  for i, msg in enumerate(state["messages"]):
    print(f" 消息 {i+1}: {type(msg).__name__} - {msg.content[:50]}{'...' if len(msg.content) > 50 else ''}")
  
  # 如果有模型则调用，否则使用模拟响应
  if model:
    try:
      response = model.invoke(state["messages"])
      print(f"生成的回复: {response.content}")
      return {"messages": [response]}
    except Exception as e:
      print(f"调用模型出错: {e}")
      # 出错时使用模拟响应
      pass
  
  # 模拟模型调用
  last_message = state["messages"][-1].content if state["messages"] else ""
  
  # 根据消息内容生成响应
  if "名字" in last_message or "name" in last_message.lower():
    response = "我记得你的名字是bob。"
  elif "诗" in last_message or "poem" in last_message.lower():
    if "猫" in last_message or "cat" in last_message.lower():
      response = "这里是一首关于猫的短诗：\\n小猫咪咪叫，\\n尾巴摇啊摇，\\n捉鼠本领高，\\n主人乐陶陶。"
    elif "狗" in last_message or "dog" in last_message.lower():
      response = "这里是一首关于狗的短诗：\\n小狗汪汪叫，\\n忠诚又可靠，\\n看家护院好，\\n人类好朋友。"
    else:
      response = "我可以为你写一首关于猫或狗的诗。"
  elif "你好" in last_message or "hi" in last_message.lower():
    response = "你好！我是AI助手。"
  else:
    response = "我理解你的问题，让我来帮助你解答。"
  
  print(f"生成的模拟回复: {response}")
  return {"messages": [AIMessage(content=response)]}

def delete_messages(state: MessagesState):
  """
  删除消息的节点函数
  
  Args:
    state: 当前状态，包含消息历史
    
  Returns:
    dict: 更新后的状态
  """
  print("\\n执行节点: delete_messages")
  messages = state["messages"]
  print(f"删除前消息数量: {len(messages)}")
  
  if len(messages) > 2:
    # 删除最早的两条消息
    to_remove = [RemoveMessage(id=m.id) for m in messages[:2]]
    print(f"将删除 {len(to_remove)} 条消息")
    # 显示要删除的消息
    for i, msg in enumerate(messages[:2]):
      print(f" 删除消息 {i+1}: {type(msg).__name__} - {msg.content[:50]}{'...' if len(msg.content) > 50 else ''}")
    return {"messages": to_remove}
  else:
    print("消息数量不足，无需删除")
    return {}

def main():
  """主函数 - 演示消息删除功能"""
  print("=== LangGraph 消息删除演示 ===\\n")
  
  # 创建检查点保存器
  checkpointer = InMemorySaver()
  
  # 构建图
  builder = StateGraph(MessagesState)
  builder.add_node(call_model)
  builder.add_node(delete_messages)
  
  # 添加边
  builder.add_edge(START, "call_model")
  builder.add_edge("call_model", "delete_messages")
  
  # 编译图
  app = builder.compile(checkpointer=checkpointer)
  
  # 配置线程ID
  config = {"configurable": {"thread_id": "1"}}
  
  # 第一次调用 - 问候
  print("1. 第一次调用 - 问候:")
  for event in app.stream(
    {"messages": [HumanMessage(content="hi! I'm bob")]},
    config,
    stream_mode="values"
  ):
    print(f"当前状态中的消息数量: {len(event['messages'])}")
    if event["messages"]:
      last_message = event["messages"][-1]
      print(f"最新消息: {type(last_message).__name__} - {last_message.content}")
  
  print("\\n" + "="*50 + "\\n")
  
  # 第二次调用 - 询问名字
  print("2. 第二次调用 - 询问名字:")
  for event in app.stream(
    {"messages": [HumanMessage(content="what's my name?")]},
    config,
    stream_mode="values"
  ):
    print(f"当前状态中的消息数量: {len(event['messages'])}")
    if event["messages"]:
      last_message = event["messages"][-1]
      print(f"最新消息: {type(last_message).__name__} - {last_message.content}")
  
  print("\\n" + "="*50 + "\\n")
  
  # 第三次调用 - 请求写诗
  print("3. 第三次调用 - 请求写诗:")
  for event in app.stream(
    {"messages": [HumanMessage(content="write a short poem about cats")]},
    config,
    stream_mode="values"
  ):
    print(f"当前状态中的消息数量: {len(event['messages'])}")
    if event["messages"]:
      last_message = event["messages"][-1]
      print(f"最新消息: {type(last_message).__name__} - {last_message.content}")
  
  print("\\n" + "="*50 + "\\n")
  
  # 第四次调用 - 请求写诗（关于狗）
  print("4. 第四次调用 - 请求写诗（关于狗）:")
  for event in app.stream(
    {"messages": [HumanMessage(content="now do the same but for dogs")]},
    config,
    stream_mode="values"
  ):
    print(f"当前状态中的消息数量: {len(event['messages'])}")
    if event["messages"]:
      last_message = event["messages"][-1]
      print(f"最新消息: {type(last_message).__name__} - {last_message.content}")
  
  print("\\n=== 演示完成 ===")

if __name__ == "__main__":
  main()


# 成功初始化百炼平台的通义大模型
# === LangGraph 消息删除演示 ===\n
# 1. 第一次调用 - 问候:
# 当前状态中的消息数量: 1
# 最新消息: HumanMessage - hi! I'm bob
# \n执行节点: call_model
# 当前消息数量: 1
#  消息 1: HumanMessage - hi! I'm bob
# 生成的回复: Hi Bob! ٩(◕‿◕｡)۶ That's a great name. I'm Qwen, and I'm really happy to meet you! I love making new friends. What would you like to chat about today? I'm pretty good at telling stories, helping with tricky problems, or we could just have a friendly chat about anything that interests you!
# 当前状态中的消息数量: 2
# 最新消息: AIMessage - Hi Bob! ٩(◕‿◕｡)۶ That's a great name. I'm Qwen, and I'm really happy to meet you! I love making new friends. What would you like to chat about today? I'm pretty good at telling stories, helping with tricky problems, or we could just have a friendly chat about anything that interests you!
# \n执行节点: delete_messages
# 删除前消息数量: 2
# 消息数量不足，无需删除
# \n==================================================\n
# 2. 第二次调用 - 询问名字:
# 当前状态中的消息数量: 3
# 最新消息: HumanMessage - what's my name?
# \n执行节点: call_model
# 当前消息数量: 3
#  消息 1: HumanMessage - hi! I'm bob
#  消息 2: AIMessage - Hi Bob! ٩(◕‿◕｡)۶ That's a great name. I'm Qwen, an...
#  消息 3: HumanMessage - what's my name?
# 生成的回复: Your name is Bob! I'm glad to meet you, Bob. 😊
# 当前状态中的消息数量: 4
# 最新消息: AIMessage - Your name is Bob! I'm glad to meet you, Bob. 😊
# \n执行节点: delete_messages
# 删除前消息数量: 4
# 将删除 2 条消息
#  删除消息 1: HumanMessage - hi! I'm bob
#  删除消息 2: AIMessage - Hi Bob! ٩(◕‿◕｡)۶ That's a great name. I'm Qwen, an...
# 当前状态中的消息数量: 2
# 最新消息: AIMessage - Your name is Bob! I'm glad to meet you, Bob. 😊
# \n==================================================\n
# 3. 第三次调用 - 请求写诗:
# 当前状态中的消息数量: 3
# 最新消息: HumanMessage - write a short poem about cats
# \n执行节点: call_model
# 当前消息数量: 3
#  消息 1: HumanMessage - what's my name?
#  消息 2: AIMessage - Your name is Bob! I'm glad to meet you, Bob. 😊
#  消息 3: HumanMessage - write a short poem about cats
# 生成的回复: In moonlit hush or golden sun,  
# A velvet shadow softly runs.
# With paws like snow and eyes that gleam—
# Two lanterns in a silent dream.

# They stretch and yawn, then leap with grace,
# A whisper in the quiet space.
# No crown they wear, yet still they reign,
# The rulers of hearth, lap, and lane.

# Oh, purr that hums like lullabies,
# The sudden spark in watchful eyes—
# They chase the dust, the beam, the fly,
# And sometimes deign to meet your "Hi."

# Though small their frame, their spirit's vast—
# A wild wind home at last.
# Cats, in their mystery, teach us this:
# To live with calm, and savor bliss.
# 当前状态中的消息数量: 4
# 最新消息: AIMessage - In moonlit hush or golden sun,
# A velvet shadow softly runs.
# With paws like snow and eyes that gleam—
# Two lanterns in a silent dream.

# They stretch and yawn, then leap with grace,
# A whisper in the quiet space.
# No crown they wear, yet still they reign,
# The rulers of hearth, lap, and lane.

# Oh, purr that hums like lullabies,
# The sudden spark in watchful eyes—
# They chase the dust, the beam, the fly,
# And sometimes deign to meet your "Hi."

# Though small their frame, their spirit's vast—
# A wild wind home at last.
# Cats, in their mystery, teach us this:
# To live with calm, and savor bliss.
# \n执行节点: delete_messages
# 删除前消息数量: 4
# 将删除 2 条消息
#  删除消息 1: HumanMessage - what's my name?
#  删除消息 2: AIMessage - Your name is Bob! I'm glad to meet you, Bob. 😊
# 当前状态中的消息数量: 2
# 最新消息: AIMessage - In moonlit hush or golden sun,
# A velvet shadow softly runs.
# With paws like snow and eyes that gleam—
# Two lanterns in a silent dream.

# They stretch and yawn, then leap with grace,
# A whisper in the quiet space.
# No crown they wear, yet still they reign,
# The rulers of hearth, lap, and lane.

# Oh, purr that hums like lullabies,
# The sudden spark in watchful eyes—
# They chase the dust, the beam, the fly,
# And sometimes deign to meet your "Hi."

# Though small their frame, their spirit's vast—
# A wild wind home at last.
# Cats, in their mystery, teach us this:
# To live with calm, and savor bliss.
# \n==================================================\n
# 4. 第四次调用 - 请求写诗（关于狗）:
# 当前状态中的消息数量: 3
# 最新消息: HumanMessage - now do the same but for dogs
# \n执行节点: call_model
# 当前消息数量: 3
#  消息 1: HumanMessage - write a short poem about cats
#  消息 2: AIMessage - In moonlit hush or golden sun,
# A velvet shadow s...
#  消息 3: HumanMessage - now do the same but for dogs
# 生成的回复: With tails that wag like sweeping brooms,  
# And thunder-paws that shake the rooms,
# They greet the world with open heart—
# A joyful, slobbery work of art.

# Their ears flop down or stand up tall,
# They answer kindness, never call.
# A nudge, a whine, a boundless leap—
# Their love runs wide, runs deep, runs steep.

# They chase the ball a hundred times,
# Through mud and snow and summer chimes.
# No task too great, no walk too far—
# They follow us like faithful stars.

# With noses cold and eyes so true,
# They sense the tears we try to stow.
# A head upon our knee will say
# More than words could ever convey.

# Oh, bark that barks at shadows, guests,
# Or simply life’s unspoken zest—
# In every leap, in every sigh,
# Dogs teach us how to love and try.
# 当前状态中的消息数量: 4
# 最新消息: AIMessage - With tails that wag like sweeping brooms,
# And thunder-paws that shake the rooms,
# They greet the world with open heart—
# A joyful, slobbery work of art.

# Their ears flop down or stand up tall,
# They answer kindness, never call.
# A nudge, a whine, a boundless leap—
# Their love runs wide, runs deep, runs steep.

# They chase the ball a hundred times,
# Through mud and snow and summer chimes.
# No task too great, no walk too far—
# They follow us like faithful stars.

# With noses cold and eyes so true,
# They sense the tears we try to stow.
# A head upon our knee will say
# More than words could ever convey.

# Oh, bark that barks at shadows, guests,
# Or simply life’s unspoken zest—
# In every leap, in every sigh,
# Dogs teach us how to love and try.
# \n执行节点: delete_messages
# 删除前消息数量: 4
# 将删除 2 条消息
#  删除消息 1: HumanMessage - write a short poem about cats
#  删除消息 2: AIMessage - In moonlit hush or golden sun,
# A velvet shadow s...
# 当前状态中的消息数量: 2
# 最新消息: AIMessage - With tails that wag like sweeping brooms,
# And thunder-paws that shake the rooms,
# They greet the world with open heart—
# A joyful, slobbery work of art.

# Their ears flop down or stand up tall,
# They answer kindness, never call.
# A nudge, a whine, a boundless leap—
# Their love runs wide, runs deep, runs steep.

# They chase the ball a hundred times,
# Through mud and snow and summer chimes.
# No task too great, no walk too far—
# They follow us like faithful stars.

# With noses cold and eyes so true,
# They sense the tears we try to stow.
# A head upon our knee will say
# More than words could ever convey.

# Oh, bark that barks at shadows, guests,
# Or simply life’s unspoken zest—
# In every leap, in every sigh,
# Dogs teach us how to love and try.
# \n=== 演示完成 ===