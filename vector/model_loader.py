import json
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


_model = None
_reranker = None
_documents = None
_chunks = None
_bm25 = None
_collection = None


def get_model():
    global _model

    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    return _model


def get_reranker():
    global _reranker

    if _reranker is None:
        print("Loading reranker...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return _reranker


def load_knowledge():

    global _documents, _chunks, _bm25

    if _documents is None:

        print("Loading knowledge chunks...")

        with open(r"C:\Users\320175878\Downloads\ops_bot\output\knowledge_chunks.json") as f:
            _chunks = json.load(f)

        _documents = [c["content"] for c in _chunks]

        tokenized_docs = [doc.lower().split() for doc in _documents]

        _bm25 = BM25Okapi(tokenized_docs)

    return _documents, _chunks, _bm25


def get_vector_collection():

    global _collection

    if _collection is None:

        print("Loading vector DB...")

        chroma_client = chromadb.PersistentClient(
            path=r"C:\Users\320175878\Downloads\ops_bot\vector_db"
        )

        _collection = chroma_client.get_collection("ops_knowledge")

    return _collection