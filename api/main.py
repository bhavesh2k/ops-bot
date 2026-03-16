from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from assistant.generate_answer import generate_answer_stream_api

app = FastAPI(title="Ops Bot API")

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