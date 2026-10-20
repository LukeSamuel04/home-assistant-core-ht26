"""The conversation platform for the Ollama integration."""

import logging
from typing import Literal, override

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers import intent  # 新增：用于构造绕过大模型的直接回复

from . import OllamaConfigEntry
from .const import DOMAIN
from .entity import OllamaBaseLLMEntity

# --- RAG 模块全局对齐导入 ---
from .rag.rag_database.storage import SQLiteStorage
from .rag.rag_database.embedding import OllamaEmbeddingEngine
from .rag.rag_database.retriever import RAGRetriever
from .rag.rag_utilities.generate_prompt import PromptGenerator
from .rag.rag_utilities.command_handler import CommandHandler  # 新增：导入命令处理引擎

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OllamaConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [OllamaConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class OllamaConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    OllamaBaseLLMEntity,
):
    """Ollama conversation agent."""

    _attr_supports_streaming = True

    def __init__(self, entry: OllamaConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )
        # 声明 command_handler 占位符，在添加到 HA 时再真正初始化
        self.command_handler: CommandHandler | None = None

    @override
    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

        # ==========================================
        # RAG 引擎初始化 (只在系统加载或重启时执行一次)
        # ==========================================
        db_path = self.hass.config.path("ollama_rag_memory.db")
        storage = SQLiteStorage(db_path=db_path)
        embedding_engine = OllamaEmbeddingEngine(
            host="http://host.docker.internal:11434"
        )
        # 实例化我们的系统控制台大管家
        self.command_handler = CommandHandler(storage, embedding_engine)

    @override
    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    @override
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    # ==========================================
    # RAG 专属拦截与注入助理方法
    # ==========================================
    async def _async_get_rules(self, text: str, user_id: str) -> list:
        """
        检索助理：获取特定用户的 RAG 规则对象列表。
        """
        try:
            # 优雅升级：直接复用 command_handler 中已经初始化的组件
            retriever = RAGRetriever(
                storage=self.command_handler.storage,
                embedding_engine=self.command_handler.embedding,
            )
            # 注入 user_id，确保只检索当前用户的记忆
            memory_items = await retriever.async_search(text, user_id=user_id)

            return memory_items or []

        except Exception as e:
            _LOGGER.error("RAG 数据库检索规则失败: %s", e)
            return []

    async def _async_handle_prompt(
        self, memory_items: list, user_input: conversation.ConversationInput
    ) -> None:
        """
        注入助理：将 MemoryItem 对象包装为提示词，提取纯文本并挂载。
        """
        if not memory_items:
            return

        try:
            rag_msg_obj = PromptGenerator.build_rag_message(memory_items)

            if not rag_msg_obj:
                return

            rag_content = ""
            if isinstance(rag_msg_obj, dict):
                rag_content = rag_msg_obj.get("content", "")
            else:
                rag_content = getattr(rag_msg_obj, "content", "")

            if rag_content:
                if user_input.extra_system_prompt:
                    user_input.extra_system_prompt = (
                        f"{user_input.extra_system_prompt}\n\n{rag_content}"
                    )
                else:
                    user_input.extra_system_prompt = rag_content

        except Exception as e:
            _LOGGER.error("RAG 系统提示词组装注入失败: %s", e)

    # ==========================================

    @override
    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""

        # ----------------------------------------------------
        # 提取当前用户的唯一标识 (核心隐私隔离点)
        # ----------------------------------------------------
        # user_input.context.user_id 是 HA 原生的身份令牌
        # 如果某些特殊调用没有 context，我们给个默认值防止报错
        current_user_id = (
            getattr(user_input.context, "user_id", None) or "system_default_user"
        )

        # ----------------------------------------------------
        # 路由一：控制台指令拦截 (短路返回，不走大模型)
        # ----------------------------------------------------
        if self.command_handler:
            # 将 user_id 穿透传递给控制台
            cmd_result = await self.command_handler.process_command(
                user_input.text, current_user_id
            )

            # 如果是控制台命令，直接将结果伪装成助手的回复抛给 UI
            if cmd_result is not None:
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_speech(cmd_result)
                return conversation.ConversationResult(
                    response=intent_response,
                    conversation_id=user_input.conversation_id,
                )

        # ----------------------------------------------------
        # 路由二 & 路由三：普通聊天状态分流
        # ----------------------------------------------------
        # 检查该用户自己是否开启了 RAG
        if self.command_handler and self.command_handler.is_rag_enabled(
            current_user_id
        ):
            # 路由三：RAG 开启状态 -> 去数据库捞该用户的记忆，生成 prompt
            memory_items = await self._async_get_rules(user_input.text, current_user_id)
            await self._async_handle_prompt(memory_items, user_input)
        else:
            # 路由二：RAG 关闭状态 -> 跳过检索，什么都不做，使用纯净提示词
            pass

        # ----------------------------------------------------
        # 原生管线继续
        # ----------------------------------------------------
        settings = {**self.entry.data, **self.subentry.data}

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                settings.get(CONF_LLM_HASS_API),
                settings.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
