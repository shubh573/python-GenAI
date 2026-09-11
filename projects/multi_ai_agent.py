# Multi AI Agentic system using LangGraph
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel,Field
from typing import Literal

class FlowState(BaseModel):
    question: str = Field(description="User Asked Question")
    category: Literal['coding', 'google_search', 'weather', 'sports'] = Field(default="google_search")
    answer: str =Field(default="")

class QuestionCategory(BaseModel):
    category: Literal['coding', 'google_search', 'weather', 'sports'] = Field(default="google_search")

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


search = GoogleSerperAPIWrapper()
tools = [search.run]

google_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a agent and can search for any question on google."
)


@tool
def get_weather(city:str):
    """I provides real time weather details for any city"""
    return f"The current temperature in {city} is 23.C"

weather_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a agent and can provide real time weather details."
)




def check_question_category(state: FlowState) -> FlowState:
    st_llm = llm.with_structured_output(QuestionCategory)
    res = st_llm.invoke(f"I want to know the category of my question, question is: {state.question}. If you are not sure then just give 'google_search' as a category")
    state.category = res.category
    return state

def route(state: FlowState) -> Literal['coding', 'google_search', 'weather']:
    return state.category

def coding_node(state: FlowState) -> FlowState:
    res = llm.invoke(f"You are a coding expert: {state.question}")
    state.answer = res.content
    return state

def weather_node(state: FlowState) -> FlowState:
    res = weather_agent.invoke({"messages":[{"role":"user","content":state.question}]})
    state.answer = res["messages"][-1].content
    return state

def google_search_node(state: FlowState) -> FlowState:
    res = google_agent.invoke({"messages": [{"role": "user", "content": state.question}]})
    state.answer = res["messages"][-1].content
    return state


graph = StateGraph(FlowState)

graph.add_node("check_question_category", check_question_category)
graph.add_node("coding", coding_node)
graph.add_node("weather", weather_node)
graph.add_node("google_search", google_search_node)

graph.add_edge(START, "check_question_category")
graph.add_edge("check_question_category", route)
graph.add_edge("coding", END)
graph.add_edge("weather", END)
graph.add_edge("google_search", END)

# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())


response = graph.invoke({"question":"What is the address of TechSimplUS"})

print(response)