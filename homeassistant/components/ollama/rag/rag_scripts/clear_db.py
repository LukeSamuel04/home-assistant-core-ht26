"""Standalone script to clear all records from the RAG SQLite database."""

import logging
import os
import sys
from pathlib import Path

# Setup logging to see progress in the terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 动态路径魔法（与 test_isolation 对齐）
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))  # rag_scripts
rag_dir = os.path.dirname(current_dir)  # rag
root_dir = os.path.dirname(rag_dir)  # ollama 目录

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from rag.rag_database.storage import SQLiteStorage
except ImportError as e:
    logger.error("❌ 导入失败：%s", e)
    sys.exit(1)


def clear_database() -> None:
    """Initialize database connection and truncate all memory items."""
    # 统一指向 HA 的 config 根目录下的数据库文件
    ha_root = Path(root_dir).parent.parent.parent
    db_path = ha_root / "config" / "ollama_rag_memory.db"

    logger.info("Connecting to SQLite Storage at: %s", db_path)
    storage = SQLiteStorage(str(db_path))

    logger.info("Executing purge operation on existing database records...")

    try:
        # 调用我们在 storage.py 里预留的“掀桌子”危险操作
        storage.clear_all()
        logger.info("Database cleared successfully! All records and sequences reset.")
    except Exception as e:
        logger.error("An error occurred while clearing the database: %s", e)


if __name__ == "__main__":
    # Run the clear process standalone
    clear_database()
