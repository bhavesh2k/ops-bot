from vector.model_loader import (
    get_model,
    get_reranker,
    load_knowledge,
    get_vector_collection
)
import numpy as np


def hybrid_search(query, k=5, retrieval_k=25):

    model = get_model()
    reranker = get_reranker()

    documents, chunks, bm25 = load_knowledge()

    collection = get_vector_collection()

    # VECTOR SEARCH
    query_embedding = model.encode(query)

    vector_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=retrieval_k
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
    )[:retrieval_k]

    keyword_docs = [documents[i] for i in top_indices]
    keyword_sources = [chunks[i]["source"] for i in top_indices]

    # COMBINE RESULTS
    def normalize(text):
        return " ".join(text.lower().split())

    def rrf_score(rank, k=60):
        return 1 / (k + rank)

    rrf_scores = {}
    doc_map = {}

    # VECTOR RESULTS
    for rank, (doc, src) in enumerate(zip(vector_docs, vector_sources)):
        key = normalize(doc)

        rrf_scores[key] = rrf_scores.get(key, 0) + rrf_score(rank)
        doc_map[key] = (doc, src["source"])

    # BM25 RESULTS
    for rank, (doc, src) in enumerate(zip(keyword_docs, keyword_sources)):
        key = normalize(doc)

        rrf_scores[key] = rrf_scores.get(key, 0) + rrf_score(rank)
        doc_map[key] = (doc, src)

    # SORT BY RRF SCORE
    ranked_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # FINAL MERGED RESULTS
    results = [doc_map[key] for key, _ in ranked_results]

    # ---------------- RERANK WITH FUSION ----------------

    # Prepare query-doc pairs
    pairs = [(query, doc) for doc, _ in results]

    # Get reranker scores
    rerank_scores = reranker.predict(pairs)

    # Normalize (sigmoid)
    rerank_scores = 1 / (1 + np.exp(-rerank_scores))


    # Combine RRF + reranker
    final_results = []

    for i, ((doc, src), rerank_score) in enumerate(zip(results, rerank_scores)):
        key = normalize(doc)

        rrf_val = rrf_scores[key]   # <-- RRF score from earlier

        # Combine scores (tunable weights)
        final_score = 0.7 * rerank_score + 0.3 * rrf_val

        # Optional light SQL boost (keep small, general)
        if "select" in doc.lower():
            final_score += 0.03

        final_results.append(((doc, src), final_score))


    # Sort final results
    final_results.sort(key=lambda x: x[1], reverse=True)


    # Return top-k directly (NO threshold)
    print(final_results[:k])

    return [(doc, src) for (doc, src), score in final_results[:k]]