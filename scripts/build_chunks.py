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

    # Replace more than 5 dots with a single dot
    text = re.sub(r'\.{5,}', '.', text)

    # Replace non-breaking spaces with space
    text = text.replace('\xa0', ' ')

    return text.strip()

# for split paragraphs > 500 chars due to exception 
def force_split(chunk, max_len=500):
    return [chunk[i:i+max_len] for i in range(0, len(chunk), max_len)]


def chunk_text(text, chunk_size=500, overlap=100):
    text = clean_text(text)
    
    # split based on this regex
    #heading_pattern = r'(?im)^(?=\s*(?:Q\d+:?|To\s+(?:Find|Get)|\d+(?:\.\d+)*\s+[A-Z]))'
    heading_pattern = r'(?im)^(?=\s*(?:Q\d+:?|To\s+(?:Find|Get)|\d+(?:\.\d+)*\.?[\s\xa0]+[A-Z]))'
    
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
            paragraphs = re.split(r'\n{1,}', section)
            current_chunk = ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                if len(current_chunk) + len(para) + 1 <= chunk_size:
                    current_chunk += para + "\n"
                else:
                    if current_chunk.strip():
                        chunk_to_add = current_chunk.strip()

                        # Add chunk
                        if len(chunk_to_add) > chunk_size:
                            split_chunks = force_split(chunk_to_add, chunk_size)
                            final_chunks.extend(split_chunks)
                            last_chunk = split_chunks[-1]
                        else:
                            final_chunks.append(chunk_to_add)
                            last_chunk = chunk_to_add

                        # 🔴 APPLY OVERLAP HERE
                        overlap_text = last_chunk[-overlap:] if overlap > 0 else ""
                        current_chunk = overlap_text + "\n" + para + "\n"
                    else:
                        current_chunk = para + "\n"

            if current_chunk.strip():
                chunk_to_add = current_chunk.strip()
                if len(chunk_to_add) > chunk_size:
                    final_chunks.extend(force_split(chunk_to_add, chunk_size))
                else:
                    final_chunks.append(chunk_to_add)

    # final safety net to ensure chunks are < 500 chars
    final_output = []
    for chunk in final_chunks:
        if len(chunk) > chunk_size:
            final_output.extend(force_split(chunk, chunk_size))
        else:
            final_output.append(chunk)

    return final_output


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