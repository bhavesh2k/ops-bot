import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create persistent vector DB
chroma_client = chromadb.PersistentClient(
    path=r"C:\Users\320175878\Downloads\ops_bot\vector_db"
)

collection = chroma_client.get_or_create_collection(
    name="ops_knowledge"
)

# Load chunks
with open(r"C:\Users\320175878\Downloads\ops_bot\output\knowledge_chunks.json") as f:
    chunks = json.load(f)

documents = []
ids = []
metadatas = []

for chunk in tqdm(chunks):
    documents.append(chunk["content"])
    ids.append(chunk["chunk_id"])
    metadatas.append({"source": chunk["source"]})

# Generate embeddings
embeddings = model.encode(documents)

# Store in DB
collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=ids
)

print("Embeddings stored locally.")