import json
import os
from extract_docs import load_documents
from tqdm import tqdm
import re

def clean_text(text):

    # Normalize quotes
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')

    # Remove excessive blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()

def chunk_text(text, chunk_size=500, overlap=100):

    # Clean extracted text
    text = clean_text(text)

    # Split only on headings like TO FIND / TO GET
    # sections = re.split(r'(?=TO\s+(?:FIND|GET))', text, flags=re.IGNORECASE)
    # sections = re.split(r'(?=\n(?:\d+(?:\.\d+)*\s+[A-Z]|[A-Z][A-Z\s]{5,}))', text)
    sections = re.split(
        r'(?im)(?=^\s*(?:To\s+(?:Find|Get)|\d+(?:\.\d+)*\s+[A-Z][A-Z\s]{2,}|[A-Z][A-Z\s]{5,})\s*$)',
        text
    )

    chunks = []

    for section in sections:

        section = section.strip()
        if not section:
            continue

        # If section is small enough keep as single chunk
        if len(section) <= chunk_size:
            chunks.append(section)

        else:
            # fallback paragraph chunking
            paragraphs = section.split("\n")
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + "\n"
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = para + "\n"
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
    return chunks


def build_knowledge_chunks():

    docs = load_documents(r"C:\Users\320175878\Downloads\ops_bot\data\raw_docs")
    knowledge_chunks = []

    for doc in tqdm(docs):
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            knowledge_chunks.append({
                "chunk_id": f"{doc['source']}_{i}",
                "source": doc["source"],
                "content": chunk
            })

    os.makedirs(r"C:\Users\320175878\Downloads\ops_bot\output", exist_ok=True)

    with open(r"C:\Users\320175878\Downloads\ops_bot\output\knowledge_chunks.json", "w") as f:
        json.dump(knowledge_chunks, f, indent=2)

    print(f"\nCreated {len(knowledge_chunks)} knowledge chunks")


if __name__ == "__main__":
    build_knowledge_chunks()