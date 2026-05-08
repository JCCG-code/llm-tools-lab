import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm_tools_lab.rag.rag_agent import text_rag_agent

# API Router Init
router = APIRouter(prefix="/rag", tags=["rag"])


# Local interfaces
class RagRequest(BaseModel):
    text: str
    model: str = "qwen3:8b"


@router.post("/query")
async def rag_query(request: RagRequest) -> str:
    try:
        return await asyncio.to_thread(text_rag_agent, request.text, request.model)
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama connection error")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
