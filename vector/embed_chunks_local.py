from pathlib import Path
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = BASE_DIR / "vector_db"
CHUNKS_FILE = BASE_DIR / "output" / "knowledge_chunks.json"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create persistent vector DB
chroma_client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)

collection = chroma_client.get_or_create_collection(
    name="ops_knowledge"
)

# Load chunks
with open(CHUNKS_FILE, encoding="utf-8") as f:
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