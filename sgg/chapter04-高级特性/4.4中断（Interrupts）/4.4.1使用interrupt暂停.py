from typing import TypedDict,Annotated
from langgraph.types import interrupt
from langgraph.graph import StateGraph
from langgraph.constants import START
from langgraph.checkpoint.memory import MemorySaver
class MyState(TypedDict):
  state_1:str
  state_2:Annotated[list,lambda x,y:x+y]

def node_1(state:MyState):

  print("entering node_1")
  # 使用中断
  res = interrupt(
    {
      "key_1":"value_1",
      "key_2":"value_2"
    }
  )
  return {"state_2":res}

graph = StateGraph(MyState)
graph.add_node(node_1)
graph.add_edge(START,"node_1")
checkpointer = MemorySaver()
graph = graph.compile(checkpointer=checkpointer)
config = {"configurable":{"thread_id":1}}
invoke_result = graph.invoke(
  {
    "state_1":"test",
    "state_2":["1"]
  },
  config=config
)
# 打印结果：[Interrupt(value={'key_1': 'value_1', 'key_2': 'value_2'}, id='d6cb4b6d0bc74b831f81861a50187c87')]
print(invoke_result['__interrupt__'])





# 恢复中断
from langgraph.types import Command
#打印结果： {'state_1': 'test', 'state_2': ['1', 'the value returned to interrupt invoke']}
result = graph.invoke(Command(resume=["the value returned to interrupt invoke"]),config=config)
print(result)


