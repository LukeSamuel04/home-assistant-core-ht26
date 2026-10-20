"""Command handler for RAG system configuration and rules management."""

import asyncio
import logging

from ..rag_database.embedding import OllamaEmbeddingEngine
from ..rag_database.models import MemoryItem
from ..rag_database.storage import SQLiteStorage

_LOGGER = logging.getLogger(__name__)


class CommandHandler:
    """Handles @Config commands for the RAG engine with user isolation."""

    def __init__(
        self, storage: SQLiteStorage, embedding_engine: OllamaEmbeddingEngine
    ) -> None:
        """Initialize the command handler with a set of active users."""
        self.storage = storage
        self.embedding = embedding_engine
        self.active_users: set[str] = set()

    def is_rag_enabled(self, user_id: str) -> bool:
        """检查特定用户是否开启了 RAG"""
        return user_id in self.active_users

    async def process_command(self, user_input: str, user_id: str) -> str | None:
        """
        Process user input to check for @Config commands.
        Returns the command execution result as a string, or None if not a command.
        """
        text = user_input.strip()

        # 拦截：如果不是指令，直接返回 None，放行给普通对话管线
        if not text.startswith("@Config"):
            return None

        # 提取真正的指令部分
        command = text[len("@Config") :].strip()

        try:
            # ==========================================
            # 1. 开关控制 (按用户隔离)
            # ==========================================
            if command == "rag on":
                self.active_users.add(user_id)
                return "⚙️ System prompt: RAG memory enhancement is now enabled."

            elif command == "rag off":
                self.active_users.discard(user_id)
                return "⚙️ System prompt: RAG memory enhancement is now disabled."

            # ==========================================
            # 2. 查看所有规则 (动态 ID 映射)
            # ==========================================
            elif command == "rule list":
                rows = await asyncio.to_thread(self.storage.fetch_by_user, user_id)
                if not rows:
                    return "📭 You have no memory rules at the moment."

                lines = ["📋 Your own memory rules:"]
                for idx, row in enumerate(rows):
                    display_id = idx + 1
                    content = row[2]
                    lines.append(f"[{display_id}] {content}")
                return "\n".join(lines)

            # ==========================================
            # 3. 新增规则
            # ==========================================
            elif command.startswith("rule add "):
                rule_text = command[len("rule add ") :].strip()
                if not rule_text:
                    return "❌ Error: Rule content cannot be empty."

                vector = await self.embedding.generate_embedding(rule_text)
                item = MemoryItem(content=rule_text, user_id=user_id)

                await asyncio.to_thread(self.storage.insert, item, vector)
                return f"✅ Rule added: {rule_text}"

            # ==========================================
            # 4. 删除指定规则 (动态映射反查)
            # ==========================================
            elif command.startswith("rule delete "):
                id_str = command[len("rule delete ") :].strip()
                try:
                    display_id = int(id_str)

                    rows = await asyncio.to_thread(self.storage.fetch_by_user, user_id)
                    if not (1 <= display_id <= len(rows)):
                        return f"❌ Error: Rule ID [{display_id}] is out of bounds. Please use @Config rule list to view IDs."

                    physical_id = rows[display_id - 1][0]

                    success = await asyncio.to_thread(
                        self.storage.delete_by_id_and_user, physical_id, user_id
                    )
                    if success:
                        return f"🗑️ Deleted rule [{display_id}]."
                    else:
                        return (
                            "❌ Error: Failed to delete rule. Please try again later."
                        )
                except ValueError:
                    return "❌ Error: The rule ID must be a number. Example: @Config rule delete 1"

            # ==========================================
            # 5. 修改/更新指定规则 (动态映射 + 向量重绘)
            # ==========================================
            elif command.startswith("rule update "):
                parts = command[len("rule update ") :].strip().split(" ", 1)

                if len(parts) < 2:
                    return "❌ Error: Incorrect format. Correct format: @Config rule update [ID] [new content]"

                id_str, new_text = parts[0], parts[1].strip()

                try:
                    display_id = int(id_str)

                    rows = await asyncio.to_thread(self.storage.fetch_by_user, user_id)
                    if not (1 <= display_id <= len(rows)):
                        return f"❌ Error: Rule ID [{display_id}] is out of bounds. Please use @Config rule list to view IDs."

                    physical_id = rows[display_id - 1][0]

                    new_vector = await self.embedding.generate_embedding(new_text)

                    success = await asyncio.to_thread(
                        self.storage.update_by_id_and_user,
                        physical_id,
                        user_id,
                        new_text,
                        new_vector,
                    )

                    if success:
                        return f"🔄 Successfully updated rule [{display_id}] to: {new_text}"
                    else:
                        return "❌ Error: Failed to update rule. Possible database concurrency issue."

                except ValueError:
                    return "❌ Error: The rule ID must be a number. Example: @Config rule update 2 Turn off lights in the evening"

            # ==========================================
            # 6. 一键清空个人规则
            # ==========================================
            elif command == "rule clear":
                await asyncio.to_thread(self.storage.clear_by_user, user_id)
                return "🧨 Your memory rules have been completely cleared."

            # ==========================================
            # 7. 未知指令处理
            # ==========================================
            else:
                return (
                    f"❌ Unknown command：{command}\n"
                    "Commands supported：\n"
                    "- @Config rag on\n"
                    "- @Config rag off\n"
                    "- @Config rule list\n"
                    "- @Config rule add [text]\n"
                    "- @Config rule update [id] [new text]\n"
                    "- @Config rule delete [id]\n"
                    "- @Config rule clear"
                )

        except Exception as err:
            _LOGGER.error("RAG Command execution failed for user %s: %s", user_id, err)
            return f"❌ Internal error occurred while executing command: {str(err)}"
