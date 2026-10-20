"""Standalone script to seed the RAG SQLite database."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 动态获取 HA 的根目录，并将其加入系统路径
# ==========================================
current_dir = Path(__file__).resolve().parent
ha_root = current_dir.parent.parent.parent.parent.parent
sys.path.insert(0, str(ha_root))

from homeassistant.components.ollama.rag.rag_database.embedding import (
    OllamaEmbeddingEngine,
)
from homeassistant.components.ollama.rag.rag_database.models import MemoryItem
from homeassistant.components.ollama.rag.rag_database.storage import SQLiteStorage

# 将数据库文件生成在 HA 的统一 config 文件夹中
DB_PATH = ha_root / "config" / "ollama_rag_memory.db"

# Define 50 diverse smart home memory items
MEMORY_ITEMS = [
    # Category: Security & Access (1-10)
    "If the front door is left open for more than 5 minutes after 10 PM, trigger the alarm.",
    "The security system should automatically arm itself in 'Away' mode when everyone leaves the geofence.",
    "Send a silent notification to my phone if motion is detected in the backyard between midnight and 5 AM.",
    "Disable the indoor security cameras automatically when a family member connects to the home Wi-Fi.",
    "If the smoke detector goes off, unlock all exterior doors and turn on all lights to 100% brightness.",
    "The garage door must automatically close if it has been open for more than 15 minutes.",
    "Flash the living room lights red if the smart lock battery drops below 10%.",
    "Keep the smart blinds closed when the security system is armed in 'Away' mode.",
    "If a window sensor is tripped while it's raining, alert me immediately.",
    "The guest PIN code for the front door is only active on weekends between 10 AM and 6 PM.",
    # Category: Climate & Environment (11-20)
    "Set the thermostat to 22 degrees Celsius during the day and 18 degrees Celsius at night.",
    "If the humidity in the bathroom exceeds 70%, turn on the exhaust fan for 20 minutes.",
    "Pause the HVAC system automatically if any exterior window is opened.",
    "Turn on the air purifier in the bedroom 30 minutes before my usual bedtime.",
    "If the outdoor temperature drops below freezing, run the heat pump to prevent frozen pipes.",
    "When the 'Movie Time' scene is activated, lower the thermostat by 2 degrees for comfort.",
    "Close the west-facing blinds automatically during summer afternoons to block solar heat.",
    "If the indoor air quality index (AQI) goes above 100, set the HVAC fan to continuous mode.",
    "The floor heating in the bathroom should pre-warm to 26 degrees at 6:30 AM every weekday.",
    "Do not run the lawn sprinklers if the weather forecast predicts rain within the next 12 hours.",
    # Category: Lighting & Ambience (21-30)
    "Gradually increase the bedroom lights to 50% brightness over 15 minutes before my morning alarm.",
    "Turn off all interior lights when the 'Goodnight' routine is triggered.",
    "When I arrive home after sunset, turn on the porch and hallway lights.",
    "Dim the dining room lights to 30% and set a warm color temperature during dinner time.",
    "The hallway motion sensor should only turn on the lights to 10% brightness between 11 PM and 6 AM.",
    "Turn on the LED strip behind the TV when the media player state changes to 'playing'.",
    "If the living room is unoccupied for 30 minutes, turn off the floor lamps to save energy.",
    "Change the RGB bulbs in the gaming room to purple and blue when the PC is turned on.",
    "The kitchen under-cabinet lighting should turn on automatically when someone enters the kitchen.",
    "Flash the desk lamp twice when a high-priority work email arrives.",
    # Category: User Preferences & Routines (31-40)
    "My favorite coffee is a double espresso, and the smart coffee maker should prepare it at 7 AM.",
    "I prefer waking up to classic rock music playing softly on the bedroom smart speaker.",
    "Never vacuum the home office during my scheduled Zoom meetings on weekday mornings.",
    "When I say 'Time to focus', play lo-fi beats and mute all smart home notifications for two hours.",
    "My preferred reading spot is the armchair; turn on the reading lamp when I sit there in the evening.",
    "I usually do yoga at 6 PM on Tuesdays; set the room temperature to 24 degrees and play ambient sounds.",
    "Always prompt me for confirmation before running the automated firmware updates.",
    "I prefer the TV volume to be capped at 40% after 9 PM to avoid waking up the kids.",
    "When I leave for work, start the robot vacuum in the living room and kitchen.",
    "Notify me via text message if the kids arrive home from school before 4 PM.",
    # Category: Appliances & Maintenance (41-50)
    "Remind me to replace the refrigerator water filter every six months.",
    "If the washing machine finishes its cycle, broadcast a message on the living room speaker.",
    "The dishwasher should only run between 1 AM and 5 AM to take advantage of off-peak electricity rates.",
    "Alert me if the water leak sensor under the kitchen sink detects any moisture.",
    "If the smart oven reaches the preheated temperature, send a push notification to my phone.",
    "The EV charger should pause charging if the total home power consumption exceeds 10 kW.",
    "Remind me to clean the robot vacuum's dustbin after every 5 cleaning cycles.",
    "If the smart plant sensor detects low soil moisture, blink the nearest smart bulb green.",
    "Turn off the smart plug for the space heater if it has been running for more than 2 hours consecutively.",
    "Send a maintenance alert if the HVAC filter differential pressure indicates it is clogged.",
]


async def seed_database() -> None:
    logger.info("Initializing SQLite Storage at: %s", DB_PATH)
    # 确保 config 文件夹存在
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    storage = SQLiteStorage(str(DB_PATH))

    logger.info("Clearing existing database records...")
    storage.clear_all()

    # 注意这里的 host，如果在 Dev Container 里且 Ollama 装在 Windows 宿主机，用 host.docker.internal
    logger.info("Initializing Ollama Embedding Engine...")
    embedding_engine = OllamaEmbeddingEngine(host="http://host.docker.internal:11434")

    logger.info(
        "Starting vectorization and injection of %d items...", len(MEMORY_ITEMS)
    )

    for i, text in enumerate(MEMORY_ITEMS, start=1):
        try:
            vector = await embedding_engine.generate_embedding(text)

            # 【修复点】：在这里加入你刚刚找到的真实 user_id
            item = MemoryItem(
                content=text,
                user_id="b248edd617b74cb4b4a4380c719f12ec",  # <--- 修改为你的真实 HA 用户 ID
                category="smart_home_rules",
                tier="hot",
            )
            inserted_id = storage.insert(item, vector)
            logger.info(
                "[%d/%d] Inserted ID %d: %s...",
                i,
                len(MEMORY_ITEMS),
                inserted_id,
                text[:50],
            )
        except Exception as e:
            logger.error("Failed to insert item %d: %s", i, e)

    logger.info("Database seeding completed successfully!")


if __name__ == "__main__":
    asyncio.run(seed_database())
