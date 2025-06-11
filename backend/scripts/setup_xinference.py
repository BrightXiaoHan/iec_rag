#!/usr/bin/env python3
"""
Xinference 设置脚本

这个脚本帮助自动设置和启动 Xinference 所需的模型
"""

import json
import os
import sys
import time
from typing import Any, Dict

import requests

# 添加 src 目录到 Python 路径
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

# Xinference 配置
XINFERENCE_HOST = os.getenv("XINFERENCE_HOST", "localhost")
XINFERENCE_PORT = os.getenv("XINFERENCE_PORT", "9997")
XINFERENCE_BASE_URL = f"http://{XINFERENCE_HOST}:{XINFERENCE_PORT}"

# 模型配置
MODELS_TO_LAUNCH = [
    {
        "model_name": "bge-base-en-v1.5",
        "model_type": "embedding",
        "description": "BGE 嵌入模型",
    },
    {
        "model_name": "bge-reranker-base",
        "model_type": "rerank",
        "description": "BGE 重排序模型",
    },
]


def check_xinference_server() -> bool:
    """检查 Xinference 服务器是否运行"""
    try:
        response = requests.get(f"{XINFERENCE_BASE_URL}/v1/models", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def list_running_models() -> Dict[str, Any]:
    """列出正在运行的模型"""
    try:
        response = requests.get(f"{XINFERENCE_BASE_URL}/v1/models")
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        return {}


def launch_model(model_name: str, model_type: str) -> bool:
    """启动指定的模型"""
    try:
        payload = {"model_name": model_name, "model_type": model_type}

        response = requests.post(
            f"{XINFERENCE_BASE_URL}/v1/models",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60,
        )

        if response.status_code == 200:
            print(f"✅ 成功启动模型: {model_name}")
            return True
        else:
            print(f"❌ 启动模型失败: {model_name}, 状态码: {response.status_code}")
            print(f"响应: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 启动模型异常: {model_name}, 错误: {e}")
        return False


def is_model_running(model_name: str, running_models: Dict) -> bool:
    """检查模型是否已经在运行"""
    if not running_models:
        return False

    for model_id, model_info in running_models.items():
        if model_info.get("model_name") == model_name:
            return True
    return False


def setup_models():
    """设置所有需要的模型"""
    print("🚀 开始设置 Xinference 模型...")

    # 检查服务器状态
    if not check_xinference_server():
        print("❌ Xinference 服务器未运行!")
        print("请先启动 Xinference 服务器:")
        print(f"xinference-local --host {XINFERENCE_HOST} --port {XINFERENCE_PORT}")
        return False

    print("✅ Xinference 服务器正在运行")

    # 获取当前运行的模型
    running_models = list_running_models()
    print(f"当前运行的模型数量: {len(running_models)}")

    # 启动需要的模型
    success_count = 0
    for model_config in MODELS_TO_LAUNCH:
        model_name = model_config["model_name"]
        model_type = model_config["model_type"]
        description = model_config["description"]

        print(f"\n📦 处理模型: {description} ({model_name})")

        if is_model_running(model_name, running_models):
            print(f"✅ 模型已在运行: {model_name}")
            success_count += 1
        else:
            print(f"🔄 启动模型: {model_name}")
            if launch_model(model_name, model_type):
                success_count += 1
                # 等待模型启动
                time.sleep(2)

    print(f"\n📊 设置完成: {success_count}/{len(MODELS_TO_LAUNCH)} 个模型成功")

    if success_count == len(MODELS_TO_LAUNCH):
        print("🎉 所有模型设置成功!")
        return True
    else:
        print("⚠️  部分模型设置失败")
        return False


def show_model_status():
    """显示模型状态"""
    print("\n📋 当前模型状态:")

    running_models = list_running_models()

    if not running_models:
        print("没有运行中的模型")
        return

    for model_id, model_info in running_models.items():
        model_name = model_info.get("model_name", "未知")
        model_type = model_info.get("model_type", "未知")
        print(f"  - {model_name} ({model_type}) [ID: {model_id}]")


def test_models():
    """测试模型功能"""
    print("\n🧪 测试模型功能...")

    try:
        # 测试嵌入模型
        from agent.milvus_search import get_xinference_embedding

        test_text = "这是一个测试文本"
        embedding = get_xinference_embedding(test_text)

        if embedding:
            print(f"✅ 嵌入模型测试成功，维度: {len(embedding)}")
        else:
            print("❌ 嵌入模型测试失败")

    except Exception as e:
        print(f"❌ 模型测试失败: {e}")


if __name__ == "__main__":
    print("Xinference 模型设置工具")
    print("=" * 50)

    try:
        # 设置模型
        if setup_models():
            # 显示状态
            show_model_status()

            # 测试功能
            test_models()

            print("\n✅ 设置完成! 现在可以使用知识库搜索功能了。")
        else:
            print("\n❌ 设置失败，请检查错误信息。")

    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 设置过程中出现错误: {e}")
        import traceback

        traceback.print_exc()