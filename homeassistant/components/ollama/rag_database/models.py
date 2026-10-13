"""Data Transfer Objects for RAG vector storage."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MemoryItem:
    content: str
    category: str = "home_assistant"
    tier: str = "hot"
    id: Optional[int] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    created_at: Optional[str] = None
