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
                    user_id TEXT NOT NULL,  -- 绑定用户
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
                "INSERT INTO memories (user_id, content, category, tier, embedding) VALUES (?, ?, ?, ?, ?)",
                (
                    memory.user_id,
                    memory.content,
                    memory.category,
                    memory.tier,
                    vec_blob,
                ),
            )
            memory.id = cursor.lastrowid
            conn.commit()
            return memory.id

    def fetch_by_user(self, user_id: str) -> list[tuple]:
        """按用户 ID 获取所有记忆"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user_id, content, category, tier, embedding, created_at FROM memories WHERE user_id = ?",
                (user_id,),
            )
            return cursor.fetchall()

    def clear_by_user(self, user_id: str) -> None:
        """按用户 ID 清空记忆"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            conn.commit()

    def delete_by_id_and_user(self, memory_id: int, user_id: str) -> bool:
        """根据指定的 ID 精确删除单条记忆，并且必须匹配 user_id 以防越权"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_by_id_and_user(
        self, memory_id: int, user_id: str, new_content: str, new_embedding: np.ndarray
    ) -> bool:
        """根据指定的 ID 精确更新记忆内容及向量，必须匹配 user_id"""
        vec_blob = new_embedding.astype(np.float32).tobytes()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE memories SET content = ?, embedding = ? WHERE id = ? AND user_id = ?",
                (new_content, vec_blob, memory_id, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear_all(self) -> None:
        """危险操作：清空全库所有用户的数据（仅供开发/测试环境或全局重置使用）"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='memories'")
            conn.commit()
