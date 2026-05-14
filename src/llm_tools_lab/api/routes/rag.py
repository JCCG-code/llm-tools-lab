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
    user_id: str | None = None
    session_id: str | None = None


@router.post("/query")
async def rag_query(request: RagRequest) -> tuple[str, list[str]]:
    try:
        return await asyncio.to_thread(
            text_rag_agent,
            request.text,
            request.model,
            use_hybrid=False,
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama connection error")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/hybrid")
async def rag_hybrid(request: RagRequest) -> tuple[str, list[str]]:
    try:
        return await asyncio.to_thread(
            text_rag_agent,
            request.text,
            request.model,
            use_hybrid=True,
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama connection error")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
