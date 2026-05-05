import ollama
from fastapi import FastAPI
from ollama import ListResponse

from llm_tools_lab.api.routes.agent import router as agent_router

# Fast API initialization
app = FastAPI(title="LLM Tools Lab", version="0.1.0")


# My routes
app.include_router(agent_router)


# General routes
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
async def models() -> ListResponse:
    return ollama.list()
