from fastapi import FastAPI


# ---- Initializations --------------------------
app = FastAPI(title="Unified LLM Gateway")


# ---- Endpoints ----------------------------------
@app.get("/health")
async def health():
    return "hola desde /health !!"
