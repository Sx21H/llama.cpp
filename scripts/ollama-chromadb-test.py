#!/usr/bin/env python3

import chromadb
import requests

OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"


def test_ollama_pipeline():
    # 1. Test Chat Generation
    print("1. Testing LLM chat generation...")
    chat_response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": "Ping test for agent node."}],
            "stream": False,
        },
        timeout=30,
    )
    chat_response.raise_for_status()
    chat_data = chat_response.json()
    print(f"   [LLM Output]: {chat_data.get('message', {}).get('content', '').strip()}")

    # 2. Test Embedding Generation (/api/embed is the current Ollama standard)
    print("2. Testing embedding generation...")
    embed_response = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": "Ella Neural Vault agent node grounding query.",
        },
        timeout=30,
    )
    embed_response.raise_for_status()
    embed_data = embed_response.json()

    # /api/embed returns a list of embeddings under "embeddings"
    embeddings = embed_data.get("embeddings", [[]])
    embedding = embeddings[0] if embeddings else []
    print(f"   [Embedding Dimension]: {len(embedding)}")

    # 3. Test ChromaDB Ingestion & Query
    print("3. Testing ChromaDB integration...")
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="test_vault",
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
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
    test_ollama_pipeline()
