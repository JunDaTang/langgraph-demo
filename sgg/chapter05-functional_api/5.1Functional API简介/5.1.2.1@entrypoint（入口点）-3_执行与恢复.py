import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter,interrupt
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
def workflow(topic:str)->dict:
  essay = write_essay(topic).result()

  is_approved = interrupt(
    {
      'essay':essay,
      'action':"Please approve/reject the essay"
    }
  )
  return {
    'essay':essay,
    'is_approved':is_approved
  }

import uuid
thread_id = str(uuid.uuid4())
config = {"configurable":{"thread_id":thread_id}}
res = workflow.invoke("cat",config=config)
print('当前res为：',res)

from langgraph.types import Command
human_review = True
resumed_result = workflow.invoke(Command(resume=human_review),config=config)
print('恢复后结果为：',resumed_result)
