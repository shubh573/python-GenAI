from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from agents import Agent, OpenAIChatCompletionsModel
import os
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY)
gemini_model = OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=gemini_client)

HOW_MANY_SEARCHES = int(os.getenv("HOW_MANY_SEARCHES",5))

INSTRUCTIONS = f"""
You are a research assistant. Given a user query, come up with a set of web searches
to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for.
"""

class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The Search term to use for web search.")

class WebSearchPlan(BaseModel):
    Searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")

planner_agent = Agent(name="Planner Agent", instructions=INSTRUCTIONS, model=gemini_model, output_type=WebSearchPlan)
