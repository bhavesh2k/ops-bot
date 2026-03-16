import ollama
from vector.hybrid_search import hybrid_search
import time


print("Warming up model...")

ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": "Hello"}],
    options={"num_predict": 1}
)

print("Model ready.\n")


# --------------------------------------------------
# PROMPT BUILDER
# --------------------------------------------------

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
2. If a SQL query exists that answers the question, return the SQL query EXACTLY as written.
3. Do NOT modify SQL queries.
4. Do NOT invent information.
5. If the answer cannot be found in the documentation, say:
"I could not find this information in the documentation."

Important:
If a SQL query appears in the documentation that answers the question, return that SQL query.
"""

    return prompt


# API STREAMING
def generate_answer_stream_api(question):
    """
    Streaming generator for FastAPI endpoint.
    Returns tokens only.
    """
    yield "Thinking...\n\n"   # send first token immediately
    
    results = hybrid_search(question, k=5)

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

    for chunk in stream:
        token = chunk["message"]["content"]
        yield token


# LOCAL DEBUG STREAMING
def generate_answer_stream_local(question):
    """
    Streaming generator for local CLI use.
    Also prints retrieved chunks and performance metrics.
    """

    total_start = time.perf_counter()

    # RETRIEVAL TIMING
    retrieval_start = time.perf_counter()

    results = hybrid_search(question, k=5)

    retrieval_end = time.perf_counter()
    retrieval_time = retrieval_end - retrieval_start

    print("\nRetrieved Chunks:")

    for i, (doc, src) in enumerate(results):
        print(f"\n--- Chunk {i+1} | Source: {src} ---\n")
        print(doc[:500])

    # PROMPT BUILD TIMING
    prompt_start = time.perf_counter()

    prompt = build_prompt(question, results)

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

    # -----------------------------
    # PERFORMANCE SUMMARY
    # -----------------------------

    print("\n\n---------------- PERFORMANCE ----------------")

    print(f"Retrieval time: {retrieval_time:.3f}s")
    print(f"Prompt build time: {prompt_time:.3f}s")

    if first_token_latency:
        print(f"LLM first token latency: {first_token_latency:.3f}s")

    print(f"LLM total generation time: {llm_total_time:.3f}s")

    print(f"Total pipeline time: {total_end - total_start:.3f}s")

    print("--------------------------------------------\n")



# --------------------------------------------------
# CLI INTERFACE
# --------------------------------------------------

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