import json
import chromadb
from vector.model_loader import get_model
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi


# Load embedding model
model = get_model()

# Load reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Load knowledge chunks
with open(r"C:\Users\320175878\Downloads\ops_bot\output\knowledge_chunks.json") as f:
    chunks = json.load(f)

documents = [c["content"] for c in chunks]


# Tokenize for keyword search
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Load vector DB
chroma_client = chromadb.PersistentClient(
    path=r"C:\Users\320175878\Downloads\ops_bot\vector_db"
)
collection = chroma_client.get_collection("ops_knowledge")

def hybrid_search(query, k=6):

    # VECTOR SEARCH 
    query_embedding = model.encode(query)

    vector_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )

    vector_docs = vector_results["documents"][0]
    vector_sources = vector_results["metadatas"][0]

    # KEYWORD SEARCH
    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    keyword_docs = [documents[i] for i in top_indices]
    keyword_sources = [chunks[i]["source"] for i in top_indices]

    # COMBINE RESULTS
    results = []

    # COMBINE + REMOVE DUPLICATES
    results = []
    seen = set()

    def normalize(text):
        return " ".join(text.lower().split())

    for doc, src in zip(vector_docs, vector_sources):
        key = normalize(doc)
        if key not in seen:
            results.append((doc, src["source"]))
            seen.add(key)

    for doc, src in zip(keyword_docs, keyword_sources):
        if doc not in seen:
            results.append((doc, src))
            seen.add(doc)

    # BOOST SQL CHUNKS
    results.sort(
        key=lambda x: 1 if "select" in x[0].lower() else 0,
        reverse=True
    )
    
    # RERANK RESULTS
    pairs = [(query, doc) for doc, _ in results]

    scores = reranker.predict(pairs)

    scored_results = list(zip(results, scores))

    scored_results.sort(key=lambda x: x[1], reverse=True)

    reranked = [r[0] for r in scored_results[:5]]

    return reranked


if __name__ == "__main__":

    question = input("Ask a question: ")
    results = hybrid_search(question)
    print("\nTop results:\n")

    for doc, source in results:

        print(f"Source: {source}")
        print(doc[:400])
        print("\n---\n")