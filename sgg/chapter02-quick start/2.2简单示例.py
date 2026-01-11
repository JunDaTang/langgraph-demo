from langgraph.graph import StateGraph, MessagesState, START, END
def this_is_a_method(params1:int,param2:int):
  print("this is a print")
# 1、定义节点函数
def mock_llm(state: MessagesState):
    """ 模拟调用LLM """
    return {"messages": [{"role": "ai", "content": "hello world"}]}

# 2、定义图
graph = StateGraph(MessagesState)

# 3、添加节点和边
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)

# 4、编译图
graph = graph.compile()

# 5、调用图
response = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
print(response)