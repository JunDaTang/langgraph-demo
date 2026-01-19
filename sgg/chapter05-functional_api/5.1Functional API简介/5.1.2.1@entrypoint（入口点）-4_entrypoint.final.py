from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint
from typing import Any
checkpointer = InMemorySaver()
@entrypoint(checkpointer=checkpointer)
def my_workflow(number: int, *, previous: Any = None) -> entrypoint.final[int, int]:
  previous = previous or 0
  # 将上次调用结果返回给调用方，将 2*number 存储到检查点中，并且在下次调用时，用作previous的值
  return entrypoint.final(value=previous, save=2 * number)

config = {
  "configurable": {
    "thread_id": "1"
  }
}

print(my_workflow.invoke(3, config)) # 0 首次调用时,previous为空
print(my_workflow.invoke(1, config)) # 6 第二次调用，previous为2*3=6，因此本次返回6