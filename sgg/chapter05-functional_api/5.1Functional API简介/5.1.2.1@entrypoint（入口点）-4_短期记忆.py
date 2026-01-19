from langgraph.checkpoint.memory import InMemorySaver
from typing import Any
checkpointer = InMemorySaver()
@entrypoint(checkpointer=checkpointer)
def my_workflow(number: int, *, previous: Any = None) -> int:
  previous = previous or 0
  return number + previous

config = {
  "configurable": {
    "thread_id": "some_thread_id"
  }
}

print(my_workflow.invoke(1, config)) # 1 首次调用时，previous为空，因此此处结果为空
print(my_workflow.invoke(2, config)) # 3 第二次调用时，previous参数会被赋值为上一次调用的结果1，因此此处返回的结果是3