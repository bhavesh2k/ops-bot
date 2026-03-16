from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from assistant.generate_answer import generate_answer_stream_api

from vector.model_loader import (
    get_model,
    get_reranker,
    load_knowledge,
    get_vector_collection
)

app = FastAPI(title="Ops Bot API")

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
    print("Shutting down Ops Bot...")


app = FastAPI(
    title="Ops Bot API",
    lifespan=lifespan
)

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def health():
    return {"status": "Ops Bot running"}

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