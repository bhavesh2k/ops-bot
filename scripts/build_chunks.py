import json
import os
from pathlib import Path
from extract_docs import load_documents
from tqdm import tqdm
import re


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DOCS_PATH = BASE_DIR / "data" / "raw_docs"
OUTPUT_PATH = BASE_DIR / "output"
CHUNKS_FILE = OUTPUT_PATH / "knowledge_chunks.json"


def clean_text(text):

    # Normalize quotes
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')

    # Remove excessive blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()

def chunk_text(text, chunk_size=500, overlap=100):
    text = clean_text(text)
    
    # Updated Splitter: 
    # 1. Added Q\d+:? to match Q1:, Q2:, etc.
    # 2. Kept your existing To Find/Get and numbered heading logic.
    heading_pattern = r'(?im)^(?=\s*(?:Q\d+:?|To\s+(?:Find|Get)|\d+(?:\.\d+)*\s+[A-Z]))'
    
    sections = re.split(heading_pattern, text)
    
    final_chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Protection Rule: Keep SQL and FAQs as single blocks
        is_protected_block = (
            ("SELECT" in section.upper() and ("TO GET" in section.upper() or "TO FIND" in section.upper())) or
            re.match(r'^Q\d+', section, re.IGNORECASE) # Protect FAQ chunks
        )
        
        if is_protected_block:
            final_chunks.append(section)
        elif len(section) <= chunk_size:
            final_chunks.append(section)
        else:
            # Standard paragraph fallback for other long text
            paragraphs = section.split("\n")
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += (para + "\n")
                else:
                    if current_chunk.strip():
                        final_chunks.append(current_chunk.strip())
                    current_chunk = (para + "\n")
            if current_chunk.strip():
                final_chunks.append(current_chunk.strip())
                
    return final_chunks


def build_knowledge_chunks():

    docs = load_documents(str(RAW_DOCS_PATH))
    knowledge_chunks = []

    for doc in tqdm(docs):
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            knowledge_chunks.append({
                "chunk_id": f"{doc['source']}_{i}",
                "source": doc["source"],
                "content": chunk
            })

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_chunks, f, indent=2)

    print(f"\nCreated {len(knowledge_chunks)} knowledge chunks")


if __name__ == "__main__":
    build_knowledge_chunks()