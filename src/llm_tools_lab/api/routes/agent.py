import asyncio
from collections.abc import AsyncIterable

import ollama
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm_tools_lab.models.agent import AgentResponse
from llm_tools_lab.services.ollama_client import run_agent, stream_agent

# API Router init
router = APIRouter(prefix="/agent", tags=["agent"])


# Local interfaces
class AgentRequest(BaseModel):
    message: str
    model: str = "qwen3:8b"


async def stream_agent_response(
    message: str, model: str = "qwen3:8b"
) -> AsyncIterable[str]:
    try:
        # Yield each token
        async for chunk in stream_agent(message, model, think=True):
            yield f"data: {chunk}\n\n"
    except ConnectionError:
        yield "data: [ERROR] Connection error\n\n"
    except ollama.ResponseError as e:
        yield f"data: [ERROR] API error: {e.error} (status: {e.status_code})\n\n"
    except Exception as e:
        yield f"data: [ERROR] Unexpected error: {e}\n\n"
    finally:
        yield "data: [DONE]\n\n"


@router.post("/stream", response_class=StreamingResponse)
async def stream(request: AgentRequest):
    return StreamingResponse(
        stream_agent_response(request.message, request.model),
        media_type="text/event-stream",
    )


@router.post("/chat")
async def chat(request: AgentRequest) -> AgentResponse:
    try:
        return await asyncio.to_thread(run_agent, request.message, request.model)
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama connection error")
    except RecursionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
