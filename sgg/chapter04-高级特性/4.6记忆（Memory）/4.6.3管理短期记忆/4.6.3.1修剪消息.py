"""
LangGraph 消息修剪演示 

展示了如何使用 trim_messages 函数来管理消息历史，
确保消息历史不会超过模型的最大上下文窗口限制。
如果环境中配置了API密钥，将使用百炼平台的通义大模型；否则使用模拟响应。
"""

import os
from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.utils import (
  trim_messages, 
  count_tokens_approximately 
)
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
import dotenv
dotenv.load_dotenv()  # 默认加载 .env

# 初始化模型
model = None
try:
  # 尝试初始化百炼平台的通义大模型
  # api_key = "替换成你的百炼平台API-KEY:sk-xxx"
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
  
  # 显示原始消息数量
  print(f"原始消息数量: {len(state['messages'])}")
  # print("原始消息内容:")
  # for msg in state["messages"]:
  #   print(f"  {msg}")
  
  # 修剪消息历史，保留最后的128个token
  messages = trim_messages( 
    state["messages"],
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=128,
    start_on="human",
    end_on=("human", "tool"),
  )
  
  # 显示修剪后的消息数量
  print(f"修剪后消息数量: {len(messages)}")
  # print("修剪后的消息内容:")
  # for msg in messages:
  #   print(f"  {msg}")
  
  # 如果有模型则调用，否则使用模拟响应
  if model:
    try:
      response = model.invoke(messages)
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

def main():
  """主函数 - 演示消息修剪功能"""
  print("=== LangGraph 消息修剪演示 (基于参考资料) ===\\n")
  
  # 创建检查点保存器
  checkpointer = InMemorySaver()
  
  # 构建图
  builder = StateGraph(MessagesState)
  builder.add_node(call_model)
  builder.add_edge(START, "call_model")
  
  # 编译图
  graph = builder.compile(checkpointer=checkpointer)
  
  # 配置线程ID
  config = {"configurable": {"thread_id": "1"}}
  
  # 第一次调用 - 问候
  print("1. 第一次调用 - 问候:")
  result1 = graph.invoke({
    "messages": [HumanMessage(content="hi, my name is bob")]
  }, config)
  print(f"回复: {result1['messages'][-1].content}")
  
  # 第二次调用 - 请求写诗（关于猫）
  print("\\n2. 第二次调用 - 请求写诗（关于猫）:")
  result2 = graph.invoke({
    "messages": [HumanMessage(content="write a short poem about cats")]
  }, config)
  print(f"回复: {result2['messages'][-1].content}")
  
  # 第三次调用 - 请求写诗（关于狗）
  print("\\n3. 第三次调用 - 请求写诗（关于狗）:")
  result3 = graph.invoke({
    "messages": [HumanMessage(content="now do the same but for dogs")]
  }, config)
  print(f"回复: {result3['messages'][-1].content}")
  
  # 第四次调用 - 询问名字
  print("\\n4. 第四次调用 - 询问名字:")
  final_response = graph.invoke({
    "messages": [HumanMessage(content="what's my name?")]
  }, config)
  print(f"回复: {final_response['messages'][-1].content}")
  
  # 模拟大量消息以展示修剪效果
  print("\\n5. 模拟大量消息以展示修剪效果:")
  # 添加大量消息
  many_messages: List[HumanMessage] = []
  for i in range(20):
    many_messages.append(HumanMessage(content=f"这是第{i+1}条测试消息，内容很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长很长"))
  
  result5 = graph.invoke({
    "messages": many_messages + [HumanMessage(content="what's my name?")]
  }, config)
  print(f"回复: {result5['messages'][-1].content}")
  
  print("\\n=== 演示完成 ===")

if __name__ == "__main__":
  main()




# 成功初始化百炼平台的通义大模型
# === LangGraph 消息修剪演示 (基于参考资料) ===\n
# 1. 第一次调用 - 问候:
# \n执行节点: call_model
# 原始消息数量: 1
# 修剪后消息数量: 1
# 生成的回复: Hi Bob! ٩(◕‿◕｡)۶ How's your day going?
# 回复: Hi Bob! ٩(◕‿◕｡)۶ How's your day going?
# \n2. 第二次调用 - 请求写诗（关于猫）:
# \n执行节点: call_model
# 原始消息数量: 3
# 修剪后消息数量: 3
# 生成的回复: Sure, Bob! Here's a little poem just for you:

# Whiskers twitch in morning light,
# Silent paws and eyes so bright.
# Stretching long on sun-warmed floors,
# Dreaming deep behind half-closed doors.

# A purr begins, a gentle hum,
# Like thunder soft, when storms are dumb.
# They leap, they nap, they bat at strings—
# Kings of boxes, queens of wings.

# With tails held high and glances sly,
# They own the world—and so say I.

# Cats, in all their quiet grace,
# Steal your heart and claim your space. 🐾
# 回复: Sure, Bob! Here's a little poem just for you:

# Whiskers twitch in morning light,
# Silent paws and eyes so bright.
# Stretching long on sun-warmed floors,
# Dreaming deep behind half-closed doors.

# A purr begins, a gentle hum,
# Like thunder soft, when storms are dumb.
# They leap, they nap, they bat at strings—
# Kings of boxes, queens of wings.

# With tails held high and glances sly,
# They own the world—and so say I.

# Cats, in all their quiet grace,
# Steal your heart and claim your space. 🐾
# \n3. 第三次调用 - 请求写诗（关于狗）:
# \n执行节点: call_model
# 原始消息数量: 5
# 修剪后消息数量: 1
# 生成的回复: Sure! Here's a heartwarming and informative piece about dogs, written in the same spirit as one might write about cats — celebrating their nature, quirks, and special bond with humans:

# 🐾 **Dogs: Humanity’s Loyal Companions Through the Ages** 🐾

# Dogs aren’t just pets — they’re family, protectors, healers, and best friends all wrapped in fur, wagging tails, and wet noses. 
# For over 15,000 years (possibly even longer), dogs have walked beside humans, evolving from wild wolves into the diverse, devoted companions we know today.

# From the tiniest Chihuahua to the mighty Great Dane, every dog carries within them an ancient instinct to belong — not to a pack of wolves, but to *us*.

# ### Why We Love Dogs
# It’s hard not to fall in love with a creature that greets you like you’ve been gone for years — even if it’s only been five minutes. Dogs offer unconditional love in its purest form. They don’t care if you had a bad day, forgot to shower, or cried during a commercial. All they see is *you*, and to them, you are everything.

# They communicate with wiggling bodies, perked ears, gentle nudges, and those soulful eyes that seem to say, “I’m here. Let’s go 
# on an adventure.”

# ### More Than Just Cute Faces
# Dogs are brilliant. They understand hundreds of words, read human emotions, and can be trained for incredible tasks:
# - Guide dogs help the visually impaired navigate the world.
# - Therapy dogs comfort patients in hospitals and schools.
# - Search-and-rescue dogs find people in disaster zones.
# - Detection dogs sniff out diseases like cancer or low blood sugar.
# - Herding dogs manage livestock with precision and skill.

# Their intelligence, loyalty, and empathy make them not just pets, but true partners in life.

# ### The Daily Joys of Dog Life
# Life with a dog is full of simple joys:
# - Morning walks where every lamppost tells a story.
# - Playtime with a squeaky toy that has seen better days (but is still the most precious object on Earth).
# - The sacred ritual of digging before lying down.
# - The dramatic sigh when you stop petting too soon.
# - And of course, the blissful chaos of zoomies at 2 a.m.

# Even their quirks — barking at the vacuum, tilting their head at strange noises, chasing their tails — remind us how wonderfully unique each dog truly is.

# ### A Bond Like No Other
# Science confirms what dog lovers have always known: being with a dog reduces stress, lowers blood pressure, and boosts happiness. The simple act of petting a dog releases oxytocin — the “love hormone” — in both species. It’s biological proof of our deep connection.

# In return, we give them safety, food, shelter, and endless belly rubs. But more than that — we give them love. And in doing so, 
# they give us purpose, laughter, and a reason to get outside, stay active, and live with a little more joy.

# ### Final Thoughts
# Dogs teach us patience, presence, and the beauty of living in the moment. They don’t hold grudges. They forgive easily. They love fiercely. And they leave paw prints on our hearts that never fade.

# So here’s to dogs — the loyal, goofy, brave, and loving souls who choose us, every single day.

# 🐶 *Woof.* 🐶

# —
# Let me know if you'd like this tailored to a specific breed, age group (like puppies or senior dogs), or theme (adoption, training, etc.)!
# 回复: Sure! Here's a heartwarming and informative piece about dogs, written in the same spirit as one might write about cats — celebrating their nature, quirks, and special bond with humans:

# 🐾 **Dogs: Humanity’s Loyal Companions Through the Ages** 🐾

# Dogs aren’t just pets — they’re family, protectors, healers, and best friends all wrapped in fur, wagging tails, and wet noses. 
# For over 15,000 years (possibly even longer), dogs have walked beside humans, evolving from wild wolves into the diverse, devoted companions we know today.

# From the tiniest Chihuahua to the mighty Great Dane, every dog carries within them an ancient instinct to belong — not to a pack of wolves, but to *us*.

# ### Why We Love Dogs
# It’s hard not to fall in love with a creature that greets you like you’ve been gone for years — even if it’s only been five minutes. Dogs offer unconditional love in its purest form. They don’t care if you had a bad day, forgot to shower, or cried during a commercial. All they see is *you*, and to them, you are everything.

# They communicate with wiggling bodies, perked ears, gentle nudges, and those soulful eyes that seem to say, “I’m here. Let’s go 
# on an adventure.”

# ### More Than Just Cute Faces
# Dogs are brilliant. They understand hundreds of words, read human emotions, and can be trained for incredible tasks:
# - Guide dogs help the visually impaired navigate the world.
# - Therapy dogs comfort patients in hospitals and schools.
# - Search-and-rescue dogs find people in disaster zones.
# - Detection dogs sniff out diseases like cancer or low blood sugar.
# - Herding dogs manage livestock with precision and skill.

# Their intelligence, loyalty, and empathy make them not just pets, but true partners in life.

# ### The Daily Joys of Dog Life
# Life with a dog is full of simple joys:
# - Morning walks where every lamppost tells a story.
# - Playtime with a squeaky toy that has seen better days (but is still the most precious object on Earth).
# - The sacred ritual of digging before lying down.
# - The dramatic sigh when you stop petting too soon.
# - And of course, the blissful chaos of zoomies at 2 a.m.

# Even their quirks — barking at the vacuum, tilting their head at strange noises, chasing their tails — remind us how wonderfully unique each dog truly is.

# ### A Bond Like No Other
# Science confirms what dog lovers have always known: being with a dog reduces stress, lowers blood pressure, and boosts happiness. The simple act of petting a dog releases oxytocin — the “love hormone” — in both species. It’s biological proof of our deep connection.

# In return, we give them safety, food, shelter, and endless belly rubs. But more than that — we give them love. And in doing so, 
# they give us purpose, laughter, and a reason to get outside, stay active, and live with a little more joy.

# ### Final Thoughts
# Dogs teach us patience, presence, and the beauty of living in the moment. They don’t hold grudges. They forgive easily. They love fiercely. And they leave paw prints on our hearts that never fade.

# So here’s to dogs — the loyal, goofy, brave, and loving souls who choose us, every single day.

# 🐶 *Woof.* 🐶

# —
# Let me know if you'd like this tailored to a specific breed, age group (like puppies or senior dogs), or theme (adoption, training, etc.)!
# \n4. 第四次调用 - 询问名字:
# \n执行节点: call_model
# 原始消息数量: 7
# 修剪后消息数量: 1
# 生成的回复: I don't know your name yet! But I'd love to learn it. Would you like to tell me? 😊
# 回复: I don't know your name yet! But I'd love to learn it. Would you like to tell me? 😊
# \n5. 模拟大量消息以展示修剪效果:
# \n执行节点: call_model
# 原始消息数量: 29
# 修剪后消息数量: 7
# 生成的回复: You haven't told me your name yet, so I don't know what it is. Could you please let me know what you'd like me to call you? 😊
# 回复: You haven't told me your name yet, so I don't know what it is. Could you please let me know what you'd like me to call you? 😊
# \n=== 演示完成 ===
