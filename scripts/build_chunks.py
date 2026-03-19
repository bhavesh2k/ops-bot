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
    
    # 1. Refined Splitter: 
    # - ONLY splits at "To Find/Get" or numbered headings (1.1, 2., etc.)
    # - Removed the generic [A-Z]{5,} which was accidentally catching 'SELECT'
    heading_pattern = r'(?im)^(?=\s*(?:To\s+(?:Find|Get)|\d+(?:\.\d+)*\s+[A-Z]))'
    
    sections = re.split(heading_pattern, text)
    
    final_chunks = []
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # 2. Protection Rule:
        # If the section contains a query (SELECT) and an instruction (To Find/Get),
        # we treat it as a single block and do NOT split it by character count.
        # This keeps the heading and the query in the same chunk.
        is_sql_block = "SELECT" in section.upper() and ("TO GET" in section.upper() or "TO FIND" in section.upper())
        
        if is_sql_block:
            final_chunks.append(section)
            
        # 3. Standard Logic for non-SQL text (like manuals or descriptions)
        elif len(section) <= chunk_size:
            final_chunks.append(section)
        else:
            # Only split if it's a long descriptive paragraph
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