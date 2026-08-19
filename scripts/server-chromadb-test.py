#!/usr/bin/env python3

# Start the two servers first, either engine works:
#
#   TensorRT-LLM (NVFP4 + paged KV cache, see scripts/trtllm/README.md):
#     scripts/trtllm/serve.sh chat      # port 8080
#     scripts/trtllm/serve.sh embed     # port 8081
#
#   llama.cpp:
#     llama-server -m <chat model>  --port 8080
#     llama-server -m <embed model> --port 8081 --embeddings --pooling mean
#
# Override the endpoints with CHAT_HOST / EMBED_HOST, and the model names with
# CHAT_MODEL / EMBED_MODEL (by default they are read back from /v1/models).

import os
from pathlib import Path

import chromadb
import requests

# both engines serve one model per process, so chat and embeddings come from
# two instances started separately
CHAT_HOST = os.environ.get("CHAT_HOST", "http://localhost:8080")
EMBED_HOST = os.environ.get("EMBED_HOST", "http://localhost:8081")
CHROMA_PATH = Path(__file__).resolve().parent / "chroma_test_db"


def resolve_model(host: str, env_var: str) -> str:
    # trtllm-serve requires "model" in the request body, llama-server ignores
    # it, so ask the server what it is serving instead of hardcoding a name
    override = os.environ.get(env_var)
    if override:
        return override

    response = requests.get(f"{host}/v1/models", timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def test_server_pipeline():
    # 1. Test Chat Generation
    print("1. Testing LLM chat generation...")
    chat_model = resolve_model(CHAT_HOST, "CHAT_MODEL")
    print(f"   [Chat Model]: {chat_model}")
    chat_response = requests.post(
        f"{CHAT_HOST}/v1/chat/completions",
        json={
            "model": chat_model,
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
    embed_model = resolve_model(EMBED_HOST, "EMBED_MODEL")
    print(f"   [Embed Model]: {embed_model}")
    embed_response = requests.post(
        f"{EMBED_HOST}/v1/embeddings",
        json={
            "model": embed_model,
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
