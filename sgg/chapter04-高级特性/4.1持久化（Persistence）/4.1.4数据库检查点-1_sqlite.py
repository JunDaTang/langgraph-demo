#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import operator
from typing import TypedDict, Annotated

from langgraph.checkpoint.sqlite import SqliteSaver # pip install langgraph-checkpoint-sqlite
from langgraph.graph import StateGraph,START,END
import sqlite3 

class MyState(TypedDict):
  messages:Annotated[list,operator.add]

def node_1(state:MyState):

  return {"messages":["abc","def"]}

def main():
	# 数据存储到sqlite_data目录下面，需要目录存在
  conn = sqlite3.connect(database="sgg/chapter04-高级特性/sqlite_data/langgraph_sqlite",check_same_thread=False) 
  memory = SqliteSaver(conn=conn) 

  builder = StateGraph(MyState)
  builder.add_node("node_1",node_1)
  builder.add_edge(START, "node_1")
  builder.add_edge("node_1", END)

  graph = builder.compile(checkpointer=memory)

  config = {"configurable": {"thread_id": "1"}}

  initial_state = graph.get_state(config)
  print(f"Initial state: {initial_state}")

  # 执行图
  result = graph.invoke({"messages":[]}, config)
  print(f"Result: {result}")

  # 查看执行后的状态
  final_state = graph.get_state(config)
  print(f"Final state: {final_state}")

  conn.close()

if __name__ == '__main__':
  main()



# Initial state: StateSnapshot(values={}, next=(), config={'configurable': {'thread_id': '1'}}, metadata=None, created_at=None, parent_config=None, tasks=(), interrupts=())
# Result: {'messages': ['abc', 'def']}
# Final state: StateSnapshot(values={'messages': ['abc', 'def']}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0f0ee1-82ce-6100-8001-22a8df595d0a'}}, metadata={'source': 'loop', 'step': 1, 'parents': {}}, created_at='2026-01-14T02:09:44.696243+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0f0ee1-82c9-62ea-8000-6fcb29afdee1'}}, tasks=(), interrupts=())