"""
LangGraph 对话总结演示

该演示展示了如何使用聊天模型来总结消息历史，而不是简单地修剪或删除消息。
这种方法可以避免在清理消息队列时丢失信息。
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict
import os
import dotenv
dotenv.load_dotenv()  # 默认加载 .env

# 定义状态类型
class SummaryState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], "messages"]
  summary: str

# 初始化模型（使用模拟模型）
model = None
summarization_model = None

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
  
  summarization_model = model.bind(max_tokens=128)
  print("成功初始化百炼平台的通义大模型")
except Exception as e:
  print(f"初始化模型失败: {e}")
  print("将使用模拟响应模式")

def summarize_conversation(messages: Sequence[BaseMessage], current_summary: str = "") -> str:
  """
  使用模型总结对话历史
  
  Args:
    messages: 消息列表
    current_summary: 当前摘要
    
  Returns:
    str: 更新后的摘要
  """
  if not messages:
    return current_summary
  
  # 如果有模型则调用，否则使用模拟摘要
  if summarization_model:
    try:
      # 构造总结提示
      summary_prompt = f"当前摘要: {current_summary}\\n\\n新对话:\\n"
      for msg in messages:
        if isinstance(msg, HumanMessage):
          summary_prompt += f"人类: {msg.content}\\n"
        elif isinstance(msg, AIMessage):
          summary_prompt += f"AI: {msg.content}\\n"
      
      summary_prompt += "\\n请提供一个简洁的摘要，包含重要的信息和上下文:"
      
      response = summarization_model.invoke([SystemMessage(content=summary_prompt)])
      return response.content
    except Exception as e:
      print(f"调用总结模型出错: {e}")
      # 出错时使用模拟摘要
      pass
  
  # 模拟摘要生成
  summary_content = " ".join([msg.content for msg in messages[-3:]]) # 取最后3条消息
  return f"对话摘要: {summary_content[:100]}..." # 简单截取前100个字符

def summarize_node(state: SummaryState):
  """
  总结节点函数
  
  Args:
    state: 当前状态
    
  Returns:
    dict: 更新后的状态
  """
  print("\\n执行节点: summarize_node")
  messages = state["messages"]
  current_summary = state.get("summary", "")
  
  print(f"当前消息数量: {len(messages)}")
  print(f"当前摘要: {current_summary}")
  
  # 如果消息数量超过阈值，进行总结
  if len(messages) > 4: # 当消息数量超过4条时进行总结
    print("消息数量超过阈值，开始总结对话历史...")
    # 取最近的几条消息进行总结
    recent_messages = messages[-4:] # 最近4条消息
    new_summary = summarize_conversation(recent_messages, current_summary)
    print(f"生成的新摘要: {new_summary}")
    
    # 返回更新后的摘要和保留最近的几条消息
    return {
      "summary": new_summary,
      "messages": messages[-2:] # 保留最近2条消息
    }
  else:
    print("消息数量未超过阈值，无需总结")
    return {"summary": current_summary}

def call_model(state: SummaryState):
  """
  调用模型的节点函数
  
  Args:
    state: 当前状态，包含消息历史和摘要
    
  Returns:
    dict: 更新后的状态
  """
  print("\\n执行节点: call_model")
  messages = state["messages"]
  summary = state.get("summary", "")
  
  print(f"当前消息数量: {len(messages)}")
  print(f"当前摘要: {summary}")
  
  # 构造包含摘要的完整上下文
  context_messages = []
  if summary:
    context_messages.append(SystemMessage(content=f"之前的对话摘要: {summary}"))
  
  context_messages.extend(messages)
  
  # 显示所有消息
  for i, msg in enumerate(context_messages):
    print(f" 消息 {i+1}: {type(msg).__name__} - {msg.content[:50]}{'...' if len(msg.content) > 50 else ''}")
  
  # 如果有模型则调用，否则使用模拟响应
  if model:
    try:
      response = model.invoke(context_messages)
      print(f"生成的回复: {response.content}")
      return {"messages": [response]}
    except Exception as e:
      print(f"调用模型出错: {e}")
      # 出错时使用模拟响应
      pass
  
  # 模拟模型调用
  last_message = messages[-1].content if messages else ""
  
  # 根据消息内容生成响应
  if "名字" in last_message or "name" in last_message.lower():
    if "bob" in last_message.lower():
      response = "我记得你的名字是bob。"
    else:
      response = "你还没有告诉我你的名字呢。"
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
  """主函数 - 演示对话总结功能"""
  print("=== LangGraph 对话总结演示 ===\\n")
  
  # 创建检查点保存器
  checkpointer = InMemorySaver()
  
  # 构建图
  builder = StateGraph(SummaryState)
  builder.add_node("summarize", summarize_node)
  builder.add_node("call_model", call_model)
  
  # 添加边
  builder.add_edge(START, "summarize")
  builder.add_edge("summarize", "call_model")
  
  # 编译图
  graph = builder.compile(checkpointer=checkpointer)
  
  # 配置线程ID
  config = {"configurable": {"thread_id": "1"}}
  
  # 第一次调用 - 问候
  print("1. 第一次调用 - 问候:")
  result1 = graph.invoke({
    "messages": [HumanMessage(content="hi, my name is bob")],
    "summary": ""
  }, config)
  print(f"回复: {result1['messages'][-1].content}")
  print(f"当前摘要: {result1.get('summary', '')}")
  
  print("\\n" + "="*50 + "\\n")
  
  # 第二次调用 - 请求写诗（关于猫）
  print("2. 第二次调用 - 请求写诗（关于猫）:")
  result2 = graph.invoke({
    "messages": [HumanMessage(content="write a short poem about cats")],
    "summary": result1.get("summary", "")
  }, config)
  print(f"回复: {result2['messages'][-1].content}")
  print(f"当前摘要: {result2.get('summary', '')}")
  
  print("\\n" + "="*50 + "\\n")
  
  # 第三次调用 - 请求写诗（关于狗）
  print("3. 第三次调用 - 请求写诗（关于狗）:")
  result3 = graph.invoke({
    "messages": [HumanMessage(content="now do the same but for dogs")],
    "summary": result2.get("summary", "")
  }, config)
  print(f"回复: {result3['messages'][-1].content}")
  print(f"当前摘要: {result3.get('summary', '')}")
  
  print("\\n" + "="*50 + "\\n")
  
  # 第四次调用 - 询问名字
  print("4. 第四次调用 - 询问名字:")
  result4 = graph.invoke({
    "messages": [HumanMessage(content="what's my name?")],
    "summary": result3.get("summary", "")
  }, config)
  print(f"回复: {result4['messages'][-1].content}")
  print(f"当前摘要: {result4.get('summary', '')}")
  
  print("\\n" + "="*50 + "\\n")
  
  # 第五次调用 - 添加更多对话以触发总结
  print("5. 第五次调用 - 添加更多对话以触发总结:")
  conversation_history = [
    HumanMessage(content="让我们聊聊天气"),
    AIMessage(content="好的，你想聊什么地区的天气？"),
    HumanMessage(content="北京的天气怎么样？"),
    AIMessage(content="我无法获取实时天气信息，但北京属于温带大陆性季风气候。"),
    HumanMessage(content="what's my name?") # 再次询问名字
  ]
  
  result5 = graph.invoke({
    "messages": conversation_history,
    "summary": result4.get("summary", "")
  }, config)
  print(f"回复: {result5['messages'][-1].content}")
  print(f"当前摘要: {result5.get('summary', '')}")
  
  print("\\n=== 演示完成 ===")

if __name__ == "__main__":
  main()




# 成功初始化百炼平台的通义大模型
# === LangGraph 对话总结演示 ===\n
# 1. 第一次调用 - 问候:
# \n执行节点: summarize_node
# 当前消息数量: 1
# 当前摘要:
# 消息数量未超过阈值，无需总结
# \n执行节点: call_model
# 当前消息数量: 1
# 当前摘要:
#  消息 1: HumanMessage - hi, my name is bob
# 生成的回复: Hi Bob! ٩(◕‿◕｡)۶ How can I assist you today?
# 回复: Hi Bob! ٩(◕‿◕｡)۶ How can I assist you today?
# 当前摘要:
# \n==================================================\n
# 2. 第二次调用 - 请求写诗（关于猫）:
# \n执行节点: summarize_node
# 当前消息数量: 1
# 当前摘要:
# 消息数量未超过阈值，无需总结
# \n执行节点: call_model
# 当前消息数量: 1
# 当前摘要:
#  消息 1: HumanMessage - write a short poem about cats
# 生成的回复: In moonlit hush they softly tread,  
# With paws of silk and eyes of thread—
# Gold needles catching starlight bright,
# Weaving shadows in the night.

# They stretch like dawn, all yawn and grace,
# A purr that warms the quiet space.
# No crown they wear, yet rule they do—
# O’er laps, o’er hearts, o’er you and you.

# With tails held high and secrets deep,
# They guard the dreams while mortals sleep.
# Small tigers with a gentle meow—
# The world’s most perfect hunters, now
# Curled in a sunbeam, fast asleep,
# Where even time moves slow and steep.
# 回复: In moonlit hush they softly tread,
# With paws of silk and eyes of thread—
# Gold needles catching starlight bright,
# Weaving shadows in the night.

# They stretch like dawn, all yawn and grace,
# A purr that warms the quiet space.
# No crown they wear, yet rule they do—
# O’er laps, o’er hearts, o’er you and you.

# With tails held high and secrets deep,
# They guard the dreams while mortals sleep.
# Small tigers with a gentle meow—
# The world’s most perfect hunters, now
# Curled in a sunbeam, fast asleep,
# Where even time moves slow and steep.
# 当前摘要:
# \n==================================================\n
# 3. 第三次调用 - 请求写诗（关于狗）:
# \n执行节点: summarize_node
# 当前消息数量: 1
# 当前摘要:
# 消息数量未超过阈值，无需总结
# \n执行节点: call_model
# 当前消息数量: 1
# 当前摘要:
#  消息 1: HumanMessage - now do the same but for dogs
# 生成的回复: Sure! Here's a heartwarming and informative piece about dogs, similar in tone and style to what one might expect when celebrating these amazing animals:

# 🐾 **Dogs: Humanity’s Faithful Companions Through the Ages** 🐾

# From the quiet comfort of a lapdog on a rainy evening to the boundless energy of a border collie chasing frisbees at the park, dogs have held a special place in human hearts for thousands of years. More than just pets, they are family members, protectors, 
# healers, and heroes.

# 🐕‍🦺 **A Bond Forged in Time**
# Archaeological evidence suggests that dogs were the first animals domesticated by humans—possibly as far back as 15,000 to 30,000 years ago. Descended from wolves, early dogs likely earned their place beside us by helping with hunting, guarding settlements, and offering companionship. Over time, this partnership blossomed into one of the most enduring relationships in the animal kingdom.

# 🐶 **A Dog for Every Heart**
# With over 340 recognized breeds (and countless lovable mixed breeds), there’s a dog for every lifestyle:
# - The loyal **German Shepherd**, serving in police and military roles.
# - The cheerful **Golden Retriever**, bringing joy to families and therapy wards.
# - The tiny but fearless **Chihuahua**, packing big personality into a small frame.
# - The dignified **Greyhound**, racing across fields with grace and speed.
# - And the ever-popular **Labrador Retriever**, America’s favorite breed for good reason.

# Each breed brings its own quirks and charms, but all share a deep desire to connect with humans.

# ❤️ **More Than Just Pets**
# Dogs enrich our lives in countless ways:
# - They reduce stress, lower blood pressure, and encourage physical activity.
# - Therapy dogs visit hospitals, schools, and disaster zones, offering comfort where words fall short.
# - Service dogs empower individuals with disabilities, providing independence and dignity.
# - Search-and-rescue dogs save lives in avalanches, earthquakes, and missing person cases.

# They don’t need grand gestures—just your presence, a kind word, or a well-timed belly rub—to feel loved and fulfilled.

# 🦴 **Lessons from Our Canine Teachers**
# Dogs live in the moment. They greet each day with enthusiasm, forgive easily, and love unconditionally. They remind us to:      
# - Be loyal.
# - Celebrate the simple joys—a walk in the woods, a game of fetch, a warm sunbeam.
# - Offer comfort without judgment.
# - Bark when something matters—but also know when to sit quietly by someone’s side.

# 🐕 **A Responsibility and a Privilege**
# Owning a dog is not just about cuddles and cute photos (though there will be plenty). It’s a commitment to care, training, health, and understanding. A happy dog is one that’s loved, exercised, mentally stimulated, and part of the family.

# So whether you’re a lifelong dog lover or considering welcoming a pup into your life, remember this:
# When you adopt a dog, you’re not just saving a life—you’re gaining a friend who will love you fiercely, every single day.       

# 🐾 In the end, perhaps it’s not us who rescued them.
# Maybe, just maybe, it’s always been the other way around.

# —

# Let me know if you'd like a version tailored for children, a specific breed spotlight, or fun facts about dogs! 🐶✨
# 回复: Sure! Here's a heartwarming and informative piece about dogs, similar in tone and style to what one might expect when celebrating these amazing animals:

# 🐾 **Dogs: Humanity’s Faithful Companions Through the Ages** 🐾

# From the quiet comfort of a lapdog on a rainy evening to the boundless energy of a border collie chasing frisbees at the park, dogs have held a special place in human hearts for thousands of years. More than just pets, they are family members, protectors, 
# healers, and heroes.

# 🐕‍🦺 **A Bond Forged in Time**
# Archaeological evidence suggests that dogs were the first animals domesticated by humans—possibly as far back as 15,000 to 30,000 years ago. Descended from wolves, early dogs likely earned their place beside us by helping with hunting, guarding settlements, and offering companionship. Over time, this partnership blossomed into one of the most enduring relationships in the animal kingdom.

# 🐶 **A Dog for Every Heart**
# With over 340 recognized breeds (and countless lovable mixed breeds), there’s a dog for every lifestyle:
# - The loyal **German Shepherd**, serving in police and military roles.
# - The cheerful **Golden Retriever**, bringing joy to families and therapy wards.
# - The tiny but fearless **Chihuahua**, packing big personality into a small frame.
# - The dignified **Greyhound**, racing across fields with grace and speed.
# - And the ever-popular **Labrador Retriever**, America’s favorite breed for good reason.

# Each breed brings its own quirks and charms, but all share a deep desire to connect with humans.

# ❤️ **More Than Just Pets**
# Dogs enrich our lives in countless ways:
# - They reduce stress, lower blood pressure, and encourage physical activity.
# - Therapy dogs visit hospitals, schools, and disaster zones, offering comfort where words fall short.
# - Service dogs empower individuals with disabilities, providing independence and dignity.
# - Search-and-rescue dogs save lives in avalanches, earthquakes, and missing person cases.

# They don’t need grand gestures—just your presence, a kind word, or a well-timed belly rub—to feel loved and fulfilled.

# 🦴 **Lessons from Our Canine Teachers**
# Dogs live in the moment. They greet each day with enthusiasm, forgive easily, and love unconditionally. They remind us to:      
# - Be loyal.
# - Celebrate the simple joys—a walk in the woods, a game of fetch, a warm sunbeam.
# - Offer comfort without judgment.
# - Bark when something matters—but also know when to sit quietly by someone’s side.

# 🐕 **A Responsibility and a Privilege**
# Owning a dog is not just about cuddles and cute photos (though there will be plenty). It’s a commitment to care, training, health, and understanding. A happy dog is one that’s loved, exercised, mentally stimulated, and part of the family.

# So whether you’re a lifelong dog lover or considering welcoming a pup into your life, remember this:
# When you adopt a dog, you’re not just saving a life—you’re gaining a friend who will love you fiercely, every single day.       

# 🐾 In the end, perhaps it’s not us who rescued them.
# Maybe, just maybe, it’s always been the other way around.

# —

# Let me know if you'd like a version tailored for children, a specific breed spotlight, or fun facts about dogs! 🐶✨
# 当前摘要:
# \n==================================================\n
# 4. 第四次调用 - 询问名字:
# \n执行节点: summarize_node
# 当前消息数量: 1
# 当前摘要:
# 消息数量未超过阈值，无需总结
# \n执行节点: call_model
# 当前消息数量: 1
# 当前摘要:
#  消息 1: HumanMessage - what's my name?
# 生成的回复: I don't know your name yet! But I'd love to learn it. Can you tell me what I should call you? 😊
# 回复: I don't know your name yet! But I'd love to learn it. Can you tell me what I should call you? 😊
# 当前摘要:
# \n==================================================\n
# 5. 第五次调用 - 添加更多对话以触发总结:
# \n执行节点: summarize_node
# 当前消息数量: 5
# 当前摘要:
# 消息数量超过阈值，开始总结对话历史...
# 生成的新摘要: 用户询问北京的天气，AI回应无法获取实时天气，但介绍了北京的气候类型。随后用户用英文问“what's my name?”，表明可能想
# 测试AI的记忆或互动能力。AI未获知用户姓名，也无法回忆此前对话中的身份信息。
# \n执行节点: call_model
# 当前消息数量: 2
# 当前摘要: 用户询问北京的天气，AI回应无法获取实时天气，但介绍了北京的气候类型。随后用户用英文问“what's my name?”，表明可能想测试AI的记忆或互动能力。AI未获知用户姓名，也无法回忆此前对话中的身份信息。
#  消息 1: SystemMessage - 之前的对话摘要: 用户询问北京的天气，AI回应无法获取实时天气，但介绍了北京的气候类型。随后用户用英...    
#  消息 2: AIMessage - 我无法获取实时天气信息，但北京属于温带大陆性季风气候。
#  消息 3: HumanMessage - what's my name?
# 生成的回复: I don't know your name. We can continue our conversation, but I won't be able to remember personal information from 
# previous interactions. How can I assist you today?
# 回复: I don't know your name. We can continue our conversation, but I won't be able to remember personal information from previous interactions. How can I assist you today?
# 当前摘要: 用户询问北京的天气，AI回应无法获取实时天气，但介绍了北京的气候类型。随后用户用英文问“what's my name?”，表明可能想测试AI的记忆或互动能力。AI未获知用户姓名，也无法回忆此前对话中的身份信息。
# \n=== 演示完成 ===