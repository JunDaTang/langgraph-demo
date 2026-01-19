from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter
from langgraph.store.base import BaseStore
from typing import Any
from langgraph.func import entrypoint,task
import time
@task
def write_essay(topic:str)->str:
  time.sleep(2)
  return f'An essay about topic:{topic}'

checkpointer = InMemorySaver()
store = InMemoryStore()
@entrypoint(checkpointer=InMemorySaver(),store=store)
def workflow(topic:str,*,previous:Any,store:BaseStore,writer:StreamWriter,config:RunnableConfig)->dict:
  print('previous打印结果为：',previous)
  print('store打印结果为：',store)
  print('writer打印结果为：',writer)
  print('config打印结果为：',config)
  essay = write_essay("cat")
  print('当前essay的类型为：',type(essay))
  print("获取结果")
  essay = essay.result()
  return {
    'essay':essay,
  }