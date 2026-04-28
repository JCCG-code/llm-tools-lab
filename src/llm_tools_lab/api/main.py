from fastapi import FastAPI


# ---- Initializations --------------------------
app = FastAPI(
    title="LLM Tools Lab",
    version="0.1.0",
    description="Progressive lab for LLM tool calling, RAG and agents",
)


# ---- Endpoints ----------------------------------
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
