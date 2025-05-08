import dotenv
import os
dotenv.load_dotenv()

os.environ["LANGSMITH_PROJECT"] = os.path.basename(__file__)
os.environ["DEEPSEEK_API_KEY"] = "sk-3cdcadd283124913bc887ab7a27734f9"


from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]

# docs[0][0].page_content.strip()[:1000]
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)


from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")
)
retriever = vectorstore.as_retriever()


from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)


retriever_tool.invoke({"query": "types of reward hacking"})


from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

response_model = init_chat_model("deepseek:deepseek-chat", temperature=0)


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        response_model
        .bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}




input = {"messages": [{"role": "user", "content": "hello!"}]}
generate_query_or_respond(input)["messages"][-1].pretty_print()


input = {
    "messages": [
        {
            "role": "user",
            "content": "What does Lilian Weng say about types of reward hacking?",
        }
    ]
}
generate_query_or_respond(input)["messages"][-1].pretty_print()