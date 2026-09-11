from dotenv import load_dotenv
load_dotenv()
import os
import requests

# from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

#### Tools
# def google_search(query:str) -> str:
#     """
#         Search Google for up-to-date information on the web
#
#         Args: query: What to search on Google
#
#         Returns: Top search results as plan texts
#     """
#
#     search = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))
#     return search.run(query)

@tool
def google_search(query: str) -> str:
    """Search Google using Serper."""

    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": os.environ["SERPER_API_KEY"],
            "Content-Type": "application/json",
        },
        json={"q": query},
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()

    snippets = []
    for result in data.get("organic", [])[:5]:
        snippets.append(
            f"{result['title']}\n{result['link']}\n{result.get('snippet', '')}"
        )

    return "\n\n".join(snippets)


### LLM = Groq

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
    You are a helpful research assistant with access to google.
"""

agent = create_agent(
    model=llm,
    tools=[google_search],
    system_prompt=SYSTEM_PROMPT
)

# result = agent.invoke({"messages":[{"role":"user", "content":"What is Agentic AI"}]},
#                       config={"configurable": {"thread_id":"1"}})
#
# print(result["messages"][-1].content)