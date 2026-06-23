import uuid
from fastapi import FastAPI, HTTPException
from src.pipeline import ask
from src.schemas import AskRequest, AskResponse, ErrorResponse

app = FastAPI(title="Pharma Diligence Copilot", version="1.0")


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    try:
        return ask(req.question, filters=req.filters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
