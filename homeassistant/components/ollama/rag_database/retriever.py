"""Retrieval engine for vector similarity search."""

import asyncio
import numpy as np
from .embedding import OllamaEmbeddingEngine
from .models import MemoryItem
from .storage import SQLiteStorage


class RAGRetriever:
    def __init__(
        self, storage: SQLiteStorage, embedding_engine: OllamaEmbeddingEngine
    ) -> None:
        self.storage = storage
        self.embedding = embedding_engine

    async def async_search(self, query: str, top_k: int = 3) -> list[MemoryItem]:
        query_vector = await self.embedding.generate_embedding(query)
        rows = await asyncio.to_thread(self.storage.fetch_all)

        if not rows:
            return []

        memories = []
        db_vectors = []

        for row in rows:
            row_id, content, category, tier, embedding_blob, created_at = row
            vec = np.frombuffer(embedding_blob, dtype=np.float32)
            db_vectors.append(vec)
            item = MemoryItem(
                id=row_id,
                content=content,
                category=category,
                tier=tier,
                embedding=vec,
                created_at=created_at,
            )
            memories.append(item)

        db_matrix = np.array(db_vectors)
        dot_products = np.dot(db_matrix, query_vector)
        query_norm = np.linalg.norm(query_vector)
        db_norms = np.linalg.norm(db_matrix, axis=1)

        db_norms[db_norms == 0] = 1e-10
        query_norm = query_norm if query_norm != 0 else 1e-10

        similarities = dot_products / (db_norms * query_norm)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [memories[i] for i in top_indices]
