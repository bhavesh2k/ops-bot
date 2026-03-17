from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from pathlib import Path

from assistant.generate_answer import generate_answer_stream_api

from vector.model_loader import (
    get_model,
    get_reranker,
    load_knowledge,
    get_vector_collection
)

BASE_DIR = Path(__file__).resolve().parents[1]
UI_DIR = BASE_DIR / "ui"

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup logic
    print("Initializing retrieval system...")

    get_model()
    get_reranker()
    load_knowledge()
    get_vector_collection()

    print("Retrieval system ready.")
    yield

    # Shutdown logic (optional)
    print("Shutting down OneSpace Service Ops Assistant...")


app = FastAPI(
    title="Ops Bot API",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def health():
    return {"status": "OneSpace Service Ops Assistant running"}


@app.get("/chat")
def chat_ui():
    return FileResponse(UI_DIR / "index.html")


# Non-streaming response
@app.post("/ask")
def ask(req: QuestionRequest):

    answer = ""
    for token in generate_answer_stream_api(req.question):
        answer += token

    return {"answer": answer}

# Streaming response
@app.post("/ask/stream")
def ask_stream(req: QuestionRequest):

    return StreamingResponse(
        generate_answer_stream_api(req.question),
        media_type="text/plain"
    )