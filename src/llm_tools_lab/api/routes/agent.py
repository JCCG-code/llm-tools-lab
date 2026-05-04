from collections.abc import AsyncIterable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm_tools_lab.services.ollama_client import stream_agent

# API Router init
router = APIRouter(prefix="/agent", tags=["agent"])


# Local interfaces
class AgentRequest(BaseModel):
    message: str
    model: str = "qwen3:8b"


async def stream_agent_response(
    message: str, model: str = "qwen3:8b"
) -> AsyncIterable[str]:
    """Async generator que hace streaming del agent loop."""
    # Yield each token
    async for chunk in stream_agent(message, model, think=True):
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/stream", response_class=StreamingResponse)
async def stream(request: AgentRequest):
    return StreamingResponse(
        stream_agent_response(request.message, request.model),
        media_type="text/event-stream",
    )
