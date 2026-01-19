from langgraph.func import entrypoint,task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt,Command
checkpointer = InMemorySaver()
@task
def slow_computation(params:list):
  import time
  print('开始执行slow_computation')
  time.sleep(5) # 此处模拟费时操作
  print("slow_computation执行结束")
  return sum(params)

@entrypoint(checkpointer=checkpointer)
def my_workflow(params:list):
  print("开始执行my_workflow")
  computation_result = slow_computation(params).result()
  user_input = interrupt(
    "中断，等待用户输入"
  )
  print(f"用户输入：{user_input}")
  print("my_workflow执行结束")
  return computation_result+user_input
config = {"configurable":{"thread_id":1}}
my_workflow.invoke([1,2,3],config=config) 
my_workflow.invoke(Command(resume=4),config=config)