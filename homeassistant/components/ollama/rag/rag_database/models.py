"""Data Transfer Objects for RAG vector storage."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MemoryItem:
    content: str
    user_id: str  # 新增：必须提供 user_id 才能创建记忆
    category: str = "home_assistant"
    tier: str = "hot"
    id: Optional[int] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    created_at: Optional[str] = None
