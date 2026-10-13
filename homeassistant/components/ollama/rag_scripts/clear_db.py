"""Standalone script to clear all records from the RAG SQLite database."""

import logging
import sys
from pathlib import Path

# Setup logging to see progress in the terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 动态获取 HA 的根目录，并将其加入系统路径
# 当前脚本路径: homeassistant/components/ollama/rag_scripts/clear_db.py
current_dir = Path(__file__).resolve().parent
ha_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(ha_root))

# 使用官方组件结构下的绝对路径导入 SQLiteStorage
from homeassistant.components.ollama.rag_database.storage import SQLiteStorage  # noqa: E402


def clear_database() -> None:
    """Initialize database connection and truncate all memory items."""
    # 统一指向 HA 的 config 根目录下的数据库文件
    db_path = str(ha_root / "config" / "ollama_rag_memory.db")

    logger.info("Connecting to SQLite Storage at: %s", db_path)
    storage = SQLiteStorage(db_path)

    logger.info("Executing purge operation on existing database records...")

    try:
        storage.clear_all()
        logger.info("Database cleared successfully! All records and sequences reset.")
    except Exception as e:  # noqa: BLE001
        logger.error("An error occurred while clearing the database: %s", e)


if __name__ == "__main__":
    # Run the clear process standalone
    clear_database()
