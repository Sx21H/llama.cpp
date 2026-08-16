#!/usr/bin/env python3

from pathlib import Path

import chromadb
import requests

# llama-server serves one model per process, so chat and embeddings come from
# two instances started separately (the embedding one needs --embeddings)
CHAT_HOST = "http://localhost:8080"
EMBED_HOST = "http://localhost:8081"
CHROMA_PATH = Path(__file__).resolve().parent / "chroma_test_db"


def test_server_pipeline():
    # 1. Test Chat Generation
    print("1. Testing LLM chat generation...")
    chat_response = requests.post(
        f"{CHAT_HOST}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Ping test for agent node."}],
            "stream": False,
        },
        timeout=30,
    )
    chat_response.raise_for_status()
    chat_data = chat_response.json()
    content = chat_data["choices"][0]["message"]["content"]
    print(f"   [LLM Output]: {content.strip()}")

    # 2. Test Embedding Generation
    print("2. Testing embedding generation...")
    embed_response = requests.post(
        f"{EMBED_HOST}/v1/embeddings",
        json={
            "input": "Ella Neural Vault agent node grounding query.",
        },
        timeout=30,
    )
    embed_response.raise_for_status()
    embed_data = embed_response.json()

    embedding = embed_data["data"][0]["embedding"]
    print(f"   [Embedding Dimension]: {len(embedding)}")

    # 3. Test ChromaDB Ingestion & Query
    print("3. Testing ChromaDB integration...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name="test_vault",
        metadata={"hnsw:space": "cosine"}
    )

    # upsert so re-runs refresh doc_1 instead of being dropped as a duplicate
    collection.upsert(
        ids=["doc_1"],
        embeddings=[embedding],
        documents=["Ella agent test document."],
    )

    results = collection.query(
        query_embeddings=[embedding],
        n_results=1
    )

    if results["documents"] and results["documents"][0]:
        print(f"   [ChromaDB Match]: {results['documents'][0][0]}")

    print("\nIntegration test successful!")


if __name__ == "__main__":
    test_server_pipeline()
