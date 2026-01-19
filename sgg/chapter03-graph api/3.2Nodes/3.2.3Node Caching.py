import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

# 定义状态
class State(TypedDict):
  x: int
  result: int

# 创建图
builder = StateGraph(State)

# 定义节点
def expensive_node(state: State) -> dict[str, int]:
  # expensive computation
  time.sleep(2)
  return {"result": state["x"] * 2}

# 添加节点，并设置缓存策略
builder.add_node("expensive_node", expensive_node, cache_policy=CachePolicy(ttl=3))
# 设置入口和出口
builder.set_entry_point("expensive_node")
builder.set_finish_point("expensive_node")

# 编译图
graph = builder.compile(cache=InMemoryCache())

# 执行图
print(graph.invoke({"x": 5}, stream_mode='updates'))
# [{'expensive_node': {'result': 10}}]
# 第二次运行利用缓存并快速返回
print(graph.invoke({"x": 5}, stream_mode='updates'))
# [{'expensive_node': {'result': 10}, '__metadata__': {'cached': True}}]