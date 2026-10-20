"""Standalone script to test RAG similarity search."""

import asyncio
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
ha_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(ha_root))

from homeassistant.components.ollama.rag_database.embedding import OllamaEmbeddingEngine
from homeassistant.components.ollama.rag_database.storage import SQLiteStorage
from homeassistant.components.ollama.rag_database.retriever import RAGRetriever

DB_PATH = ha_root / "config" / "ollama_rag_memory.db"


async def test_search():
    print("Initializing components for search test...")
    storage = SQLiteStorage(str(DB_PATH))
    embedding_engine = OllamaEmbeddingEngine(host="http://host.docker.internal:11434")
    retriever = RAGRetriever(storage, embedding_engine)

    # 模拟用户的一个提问
    query = "What should I do if the front door is left open late at night?"
    print(f"\nTest Query: '{query}'")

    # 执行检索，获取 Top-2 最相关的记忆
    results = await retriever.async_search(query, top_k=2)

    print("\nTop Matching Results found:")
    for i, item in enumerate(results, start=1):
        print(f"[{i}] Category: {item.category} | Content: {item.content}")


if __name__ == "__main__":
    asyncio.run(test_search())
