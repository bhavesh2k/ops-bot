import chromadb
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load persistent vector DB
chroma_client = chromadb.PersistentClient(
    path=r"C:\Users\320175878\Downloads\ops_bot\vector_db"
)

collection = chroma_client.get_collection("ops_knowledge")


def search(query, k=3):

    embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=k
    )

    return results


if __name__ == "__main__":

    question = input("Ask a question: ")

    results = search(question)

    docs = results["documents"][0]
    sources = results["metadatas"][0]

    print("\nTop results:\n")

    for i in range(len(docs)):

        print(f"Source: {sources[i]['source']}")
        print(docs[i][:500])
        print("\n---\n")