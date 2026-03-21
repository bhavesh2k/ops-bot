from pathlib import Path
from vector.hybrid_search import hybrid_search
import json
import ollama
import time
import datetime
import os

# warm up model
print("Warming up model...")
ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": "Hello"}],
    options={"num_predict": 1}
)

print("Model ready.\n")


# prompt builder
def build_prompt(question, results):

    context = "\n\n".join([doc for doc, src in results])

    prompt = f"""
You are an internal operations assistant.

You must answer ONLY using the documentation provided below.

Documentation:
{context}

Question:
{question}

Rules:
1. If the answer exists in the documentation, extract it directly.
2. Prefer the chunk that most directly answers the question.
3. If a SQL query exists that answers the question, return the SQL query EXACTLY as written.
4. Do NOT modify SQL queries.
5. Do NOT invent information.
6. If one chunk answers the question, just use that. DO NOT combine chunks.
7. If the answer cannot be found in the documentation, say:
"I could not find this information in the documentation."

Important:
If a SQL query appears in the documentation that answers the question, return that SQL query.
"""
    return prompt


# rewrite query for better retrieval
def rewrite_query(question):
    """
    Rewrite user query to improve retrieval.
    Keeps it short but keyword rich.
    """

    rewrite_prompt = f"""
Rewrite the following user query to improve document retrieval.

Make it:
- more explicit
- keyword rich
- suitable for searching technical documentation or SQL examples

User Query:
{question}

Rewritten Query:
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": rewrite_prompt}],
        options={
            "num_predict": 40,
            "temperature": 0
        }
    )

    rewritten = response["message"]["content"].strip()

    # fallback safety
    if not rewritten:
        return question

    return rewritten


# logging helper function
def log_query(question, status, sources):

    BASE_DIR = Path(__file__).resolve().parents[1]
    log_dir = BASE_DIR / "logs"
    log_file = os.path.join(log_dir, "queries.log")

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    source_str = ", ".join(sources) if sources else "none"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {question} | {status} | {source_str}\n")


# API STREAMING
def generate_answer_stream_api(question):
    """
    Streaming generator for FastAPI endpoint.
    Returns tokens only.
    """
    yield json.dumps({
        "type": "thinking"
    }) + "\n"

    answer_text = ""
    # rewritten_query = rewrite_query(question)
    results = hybrid_search(question, k=5)  # result contains [(doc, source)]

    # handle cases where retrieval finds nothing
    if not results:
        yield '{"type":"sources","data":[]}\n'
        yield '{"type":"start_answer"}\n'
        yield '{"type":"token","data":"I could not find this information in the documentation."}\n'
        return

     # Extract unique sources
    sources = list(dict.fromkeys([src for doc, src in results]))
    
    # send sources as structured event
    yield json.dumps({
        "type": "sources",
        "data": sources
    }) + "\n"

    # start answer stream
    yield json.dumps({
        "type": "start_answer"
    }) + "\n"

    prompt = build_prompt(question, results)

    stream = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={
            "num_predict": 300,
            "temperature": 0.2
        }
    )

    first_token = True

    for chunk in stream:
        token = chunk["message"]["content"]
        
        if first_token:
            token = token.lstrip()  # remove leading whitespace from the first token
            first_token = False
        
        answer_text += token
        yield json.dumps({
            "type": "token",
            "data": token
        }) + "\n"
    
    # detect and log unanswered questions
    if "I could not find this information" in answer_text:
        status = "not_found"
    else:
        status = "answered"

    log_query(question, status, sources)


# local debug streaming
def generate_answer_stream_local(question):
    """
    Streaming generator for local CLI use.
    Also prints retrieved chunks and performance metrics.
    """

    total_start = time.perf_counter()

    # RETRIEVAL TIMING
    retrieval_start = time.perf_counter()

    # rewrite query and then retreive
    # rewritten_query = rewrite_query(question)
    # print(f"\nRewritten query: {rewritten_query}\n")
    results = hybrid_search(question, k=5)

    if not results:
        print('Sources - NA\n')
        print('I could not find this information in the documentation.\n')        
        return

    retrieval_end = time.perf_counter()
    retrieval_time = retrieval_end - retrieval_start

    '''print("\nRetrieved Chunks:")

    for i, (doc, src) in enumerate(results):
        print(f"\n--- Chunk {i+1} | Source: {src} ---\n")
        print(doc[:500])'''

    # PROMPT BUILD TIMING
    prompt_start = time.perf_counter()

    prompt = build_prompt(question, results)
    # print(prompt)

    prompt_end = time.perf_counter()
    prompt_time = prompt_end - prompt_start

    # LLM INFERENCE
    llm_start = time.perf_counter()

    stream = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={
            "num_predict": 300,
            "temperature": 0.2
        }
    )

    first_token_time = None
    first_token = True
    answer = ""

    print("\nAnswer:\n")

    for chunk in stream:

        if first_token_time is None:
            first_token_time = time.perf_counter()

        token = chunk["message"]["content"]

        if first_token:
            token = token.lstrip()
            first_token = False

        print(token, end="", flush=True)
        answer += token
        yield token

    llm_end = time.perf_counter()

    llm_total_time = llm_end - llm_start
    first_token_latency = first_token_time - llm_start if first_token_time else None

    total_end = time.perf_counter()

    # PERFORMANCE SUMMARY
    print("\n\n---------------- PERFORMANCE ----------------")
    print(f"Retrieval time: {retrieval_time:.3f}s")
    print(f"Prompt build time: {prompt_time:.3f}s")

    if first_token_latency:
        print(f"LLM first token latency: {first_token_latency:.3f}s")

    print(f"LLM total generation time: {llm_total_time:.3f}s")

    print(f"Total pipeline time: {total_end - total_start:.3f}s")
    print("--------------------------------------------\n")


# CLI INTERFACE
if __name__ == "__main__":

    print("Welcome to OneSpace Service Ops Assistant!\n")

    while True:
        question = input("Ask OneSpace Service Ops Assistant (type 'bye' to exit): ").strip()

        if question.lower() == "bye":
            print("Goodbye!")
            break

        print("\nGenerating answer...\n")

        for _ in generate_answer_stream_local(question):
            pass