import asyncio
import os
import sys

# 【核心修改】：动态获取当前脚本所在目录，并向上回退两级找到 "ollama" 目录（也就是 rag 文件夹的父目录）
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前是 rag_scripts
rag_dir = os.path.dirname(current_dir)  # 上级是 rag
root_dir = os.path.dirname(rag_dir)  # 再上级是包含 rag 的根目录

# 将根目录注入到 Python 的临时环境变量中，这样就能顺畅地 import rag 开头的包了
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from rag.rag_database.storage import SQLiteStorage
    from rag.rag_database.embedding import OllamaEmbeddingEngine
    from rag.rag_utilities.command_handler import CommandHandler
except ImportError as e:
    print(f"❌ 导入失败：{e}")
    sys.exit(1)


async def main():
    # 1. 准备真实的系统组件（创建真实的测试数据库文件）
    db_path = "test_isolation_memory.db"

    # 每次跑测试前，确保是个干净的环境
    if os.path.exists(db_path):
        os.remove(db_path)

    print("🛠️ 初始化系统核心组件...")
    # 真实连接本地 SQLite
    storage = SQLiteStorage(db_path=db_path)
    # 真实连接本地的 Ollama 服务 (请确保你的 Ollama 正在运行)
    embedding_engine = OllamaEmbeddingEngine(host="http://host.docker.internal:11434")
    # 真实实例化大管家
    handler = CommandHandler(storage=storage, embedding_engine=embedding_engine)

    # 模拟 Home Assistant 底层解析出的两个不同用户的 ID
    user_alice = "ha_user_alice_111"
    user_bob = "ha_user_bob_222"

    print("\n" + "=" * 60)
    print("👩 账号 A (Alice) 模拟登录并操作")
    print("=" * 60)

    # Alice 添加她的专属规则
    print("▶️ Alice 正在添加规则...")
    res = await handler.process_command(
        "@Config rule add 我非常怕冷，空调永远开 28 度", user_alice
    )
    print(f"系统返回: {res}")

    res = await handler.process_command(
        "@Config rule add 我每天晚上 11 点准时睡觉", user_alice
    )
    print(f"系统返回: {res}")

    # Alice 查看列表
    print("\n▶️ Alice 查看她的规则列表：")
    res = await handler.process_command("@Config rule list", user_alice)
    print(res)

    print("\n" + "=" * 60)
    print("👨 账号 B (Bob) 模拟登录并操作")
    print("=" * 60)

    # Bob 刚登录，查看列表（绝密隔离测试：他能不能看到 Alice 的 28 度空调？）
    print("▶️ Bob 刚登录，查看他的规则列表：")
    res = await handler.process_command("@Config rule list", user_bob)
    print(res)

    # Bob 添加他自己的规则
    print("\n▶️ Bob 正在添加规则...")
    res = await handler.process_command(
        "@Config rule add 我喜欢蓝色，客厅灯设为蓝色", user_bob
    )
    print(f"系统返回: {res}")

    # Bob 再次查看列表
    print("\n▶️ Bob 添加后查看他的规则列表：")
    res = await handler.process_command("@Config rule list", user_bob)
    print(res)

    print("\n" + "=" * 60)
    print("👩 账号 A (Alice) 再次查看")
    print("=" * 60)
    # Alice 再次查看规则（绝密隔离测试：Bob 加的蓝色灯光有没有串门过来？）
    print("▶️ Alice 再次查看她的规则列表：")
    res = await handler.process_command("@Config rule list", user_alice)
    print(res)

    print("\n" + "=" * 60)
    print("🗑️ 清理测试战场")
    print("=" * 60)
    # 测试完毕，调用我们刚写的危险方法销毁测试数据
    storage.clear_all()
    if os.path.exists(db_path):
        os.remove(db_path)
    print("✅ 测试数据库已安全销毁。")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
