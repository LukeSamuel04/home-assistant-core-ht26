"""SQLite storage operations for RAG vector data."""

import sqlite3
import numpy as np
from .models import MemoryItem


class SQLiteStorage:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    category TEXT,
                    tier TEXT,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def insert(self, memory: MemoryItem, embedding_vector: np.ndarray) -> int:
        vec_blob = embedding_vector.astype(np.float32).tobytes()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO memories (content, category, tier, embedding) VALUES (?, ?, ?, ?)",
                (memory.content, memory.category, memory.tier, vec_blob),
            )
            memory.id = cursor.lastrowid
            conn.commit()
            return memory.id

    def fetch_all(self) -> list[tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, content, category, tier, embedding, created_at FROM memories"
            )
            return cursor.fetchall()

    def clear_all(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='memories'")
            conn.commit()
