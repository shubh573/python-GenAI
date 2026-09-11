from fastapi import FastAPI
from langserve import add_routes
from agents import agent

app = FastAPI(
    title="Google Search Agent"
)

add_routes(
    app,
    agent,
    path="/search-agent",
)