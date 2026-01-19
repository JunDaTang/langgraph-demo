from typing import TypedDict
from langgraph.graph import StateGraph,START
from langchain.chat_models import init_chat_model
import os
import dotenv
dotenv.load_dotenv()  # 默认加载 .env

# model = init_chat_model(model="gpt-4o-mini",model_provider="openai")
# model = init_chat_model(
#   model="gpt-4o-mini",
#   model_provider="openai",
#   base_url=os.getenv("OPENAI_BASE_URL"),
#   api_key=os.getenv("OPENAI_API_KEY"),
# )
print("base_url:",os.getenv("DEEPSEEK_BASE_URL"))
print("api_key:",os.getenv("DEEPSEEK_API_KEY"))

# 定义模型
model = init_chat_model(
  model="deepseek-chat",
  model_provider="deepseek",
  base_url=os.getenv("DEEPSEEK_BASE_URL"),
  api_key=os.getenv("DEEPSEEK_API_KEY"),
)

class State(TypedDict):
  query:str
  answer:str

def node(state:State):
  print("开始调用node节点")
  llm_result = model.invoke(
    [("user",state["query"])]
  )
  print("llm invoke结束")
  return {"answer":llm_result}

def main():
  graph = (
    StateGraph(
      state_schema=State
    )
    .add_node(node)
    .add_edge(START,"node")
    .compile()
  )
  inputs = {"query":"帮我生成一个300字的小学生作文，主题为我的一天"}
#

  for chunk,meta_data in graph.stream(inputs,stream_mode="messages"):
    # print(f"type of chunk:{type(chunk)}")
    print(chunk.content,end="")

if __name__ == '__main__':
  main()




# 开始调用node节点
# ## 我的一天

# 清晨六点，闹钟还没响，我就醒了。因为今天，我要做一件特别的事。

# 厨房里飘来煎蛋的香味。我悄悄走到妈妈身后，她正踮着脚够柜子里的酱油。“妈妈，让我来！”我搬来小凳子站上去，刚好能够到。妈妈惊讶地回
# 头，晨光给她鬓角的白发镶了道金边。

# 原来，我已经比妈妈高了。

# 上午的数学课，老师让用尺子量东西。我量了铅笔盒、课本，最后偷偷量了同桌小胖的手掌。他咯咯直笑：“你量这个干嘛？”我没告诉他——我想记
# 住好朋友手掌的大小。

# 放学时下雨了。校门口，家长们举着伞像一片蘑菇林。我在伞下看见一双熟悉的旧皮鞋——是爷爷！他裤脚湿透了，却把伞全倾向我这边。雨水顺着
# 伞骨流成小瀑布，爷爷哼起走调的歌。那一刻，我觉得雨声是世界上最好听的音乐。

# 晚上写日记时，我画了三幅画：妈妈的白发、小胖的手掌、爷爷的伞。原来，长大就是发现那些一直爱我的人，正在慢慢变老。而我的任务，是记
# 住今天，记住每一个让我长高的瞬间。

# 星星出来了。明天，我要更早起床，给妈妈拿酱油。llm invoke结束