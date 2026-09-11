from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel,Field

# from IPython.display import Image


## Document Load
loader = PyPDFDirectoryLoader("../data/Profile.pdf")
docs = loader.load()

## split - chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(docs)

## embeddings and Vector DB
embed = OpenAIEmbeddings(model="text-embedding-3-large")
vectore_store = InMemoryVectorStore.from_documents(
    documents=docs,
    embeddings=embed
)

#llm = ChatGroq(model="openai/gpt-oss-20b")
llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


class RagState(BaseModel):
    question: str = Field(description="User Question")
    documents: list = []
    context: str = Field(description="Context data for user question", default="")
    answer: str = Field(description="Final Answer...", default="")


# question -> retrieve -> context -> generate -> end

def retrieve_node(state:RagState) -> RagState:
    docs = vectore_store.similarity_search(query=state.question)
    state.documents = docs
    return state

def create_context_node(state:RagState) -> RagState:
    context = ""
    for doc in state.documents:
        context += doc.page_content + "\n\n"
    state.context = context
    return state

def generate_node(state:RagState) -> RagState:
    prompt = f"""
        You are a assistant and provide answer for user question based on the provided context. 
        If you dont find the relevant answer then just say 'I dont know.'.
        Context is: {state.context},
        Question is: {state.question}
    """
    res = llm.invoke(prompt)
    state.answer = res.content
    return state


graph = StateGraph(RagState)

graph.add_node("retrieve_node", retrieve_node)
graph.add_node("create_context_node", create_context_node)
graph.add_node("generate_node", generate_node)

graph.add_edge(START, "retrieve_node")
graph.add_edge("retrieve_node", "create_context_node")
graph.add_edge("create_context_node", "generate_node")
graph.add_edge("generate_node", END)

graph = graph.compile()
# Image(graph.get_graph().draw_mermaid_png())

res = graph.invoke({"question":"What are the qualifications?"})
print(res["answer"])