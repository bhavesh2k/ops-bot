from vector.model_loader import (
    get_model,
    get_reranker,
    load_knowledge,
    get_vector_collection
)


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

        key = normalize(doc)

        if key not in seen:
            results.append((doc, src))
            seen.add(key)


    # RERANK RESULTS
    pairs = [(query, doc) for doc, _ in results]

    scores = reranker.predict(pairs)
    scored_results = []
    for (doc, src), score in zip(results, scores):

        # small SQL boost
        if all(keyword in doc.lower() for keyword in ("select", "from", "where")):
            score += 0.05

        scored_results.append(((doc, src), score))

    scored_results.sort(key=lambda x: x[1], reverse=True)

    # add relevance threshold
    THRESHOLD = 0.3
    filtered = [
        (doc, src)
        for (doc, src), score in scored_results
        if score >= THRESHOLD
    ]

    return filtered[:k]