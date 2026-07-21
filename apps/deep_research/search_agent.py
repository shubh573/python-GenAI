import os
from dotenv import load_dotenv

from openai import AsyncOpenAI
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    function_tool,
)

from duckduckgo_search import DDGS

load_dotenv(override=True)

MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

gemini_model = OpenAIChatCompletionsModel(
    model=MODEL_NAME,
    openai_client=client,
)


@function_tool
def web_search(query: str) -> str:
    """Search the web and return the top results."""

    results = []

    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=5):
            results.append(
                f"""
Title: {item.get('title')}
URL: {item.get('href')}
Snippet: {item.get('body')}
"""
            )

    if not results:
        return "No search results found."

    return "\n\n".join(results)

INSTRUCTIONS = """
You are a research assistant. Given a search term, you search the web for that term and
produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 words
Capture the main points and succinct. Reply only with the summary.
"""


search_agent = Agent(
    name="Search Agent",
    instructions=INSTRUCTIONS,
    model=gemini_model,
    tools=[web_search],
    model_settings=ModelSettings(tool_choice="required"),
)


## OpenAI-hosted tool
# from agents import Agent, WebSearchTool, ModelSettings
# from dotenv import load_dotenv
# import os
# 
# load_dotenv(override=True)
# MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "gpt-5.4-mini")
# 
# INSTRUCTIONS = """
# You are a research assistant. Given a search term, you search the web for that term and 
# produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 words.
# Capture the main points and be succinct. Reply only with the summary.
# """
# 
# settings = ModelSettings(tool_choice="required")
# tools = [WebSearchTool()]
# 
# search_agent = Agent(name="Search Agent", instructions=INSTRUCTIONS, tools=tools, model=MODEL_NAME, model_settings=settings)
