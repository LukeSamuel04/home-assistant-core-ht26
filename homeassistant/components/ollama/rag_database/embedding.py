"""Async embedding engine using Ollama."""

import numpy as np
from ollama import AsyncClient


class OllamaEmbeddingEngine:
    def __init__(
        self, host: str = "http://localhost:11434", model: str = "nomic-embed-text"
    ) -> None:
        self.client = AsyncClient(host=host)
        self.model = model

    async def generate_embedding(self, text: str) -> np.ndarray:
        response = await self.client.embeddings(model=self.model, prompt=text)
        vector_data = response["embedding"]
        return np.array(vector_data, dtype=np.float32)
