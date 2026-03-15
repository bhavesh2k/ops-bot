# Ops Documentation Bot

A retrieval-based bot that answers operational queries using internal documentation.

## Features

- Document ingestion
- Chunking pipeline
- Embedding generation
- Semantic search
- Query answering

## Project Structure

extract_docs.py – document loading
build_chunks.py – document chunking
embed_chunks_local.py - store the chunks in chromaDB
generate_answer.py – question answering

## Setup

pip install -r requirements.txt

## Run

python build_chunks.py
python embed_chunks_local.py
python generate_answer.py