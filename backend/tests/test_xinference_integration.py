#!/usr/bin/env python3
"""
测试 Xinference 集成功能的脚本
"""

import os
import sys

# 添加 src 目录到 Python 路径
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from agent.milvus_search import (
    check_milvus_connection,
    check_xinference_connection,
    get_xinference_embedding,
    launch_xinference_models,
)
from agent.utils import search_by_data_source, search_milvus


def test_xinference_connection():
    """测试 Xinference 连接"""
    print("=== 测试 Xinference 连接 ===")

    try:
        is_connected = check_xinference_connection()
        print(f"Xinference 连接状态: {'可用' if is_connected else '不可用'}")

        if not is_connected:
            print("尝试启动 Xinference 模型...")
            launch_xinference_models()

    except Exception as e:
        print(f"Xinference 连接测试失败: {e}")


def test_xinference_embedding():
    """测试 Xinference 嵌入功能"""
    print("\n=== 测试 Xinference 嵌入功能 ===")

    test_text = "这是一个测试文本"

    try:
        embedding = get_xinference_embedding(test_text)

        if embedding:
            print(f"文本: {test_text}")
            print(f"嵌入维度: {len(embedding)}")
            print(f"嵌入向量前5个值: {embedding[:5]}")
        else:
            print("嵌入生成失败")

    except Exception as e:
        print(f"嵌入测试失败: {e}")


def test_milvus_connection():
    """测试 Milvus 连接"""
    print("\n=== 测试 Milvus 连接 ===")

    try:
        is_connected = check_milvus_connection()
        print(f"Milvus 连接状态: {'可用' if is_connected else '不可用'}")

    except Exception as e:
        print(f"Milvus 连接测试失败: {e}")


def test_knowledge_base_search():
    """测试知识库搜索（使用 Xinference）"""
    print("\n=== 测试知识库搜索 ===")

    query = "IEC 61850 通信协议"

    try:
        results = search_by_data_source(query, "knowledge_base", 3)

        print(f"查询: {query}")
        print(f"结果数量: {len(results)}")

        for i, result in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(f"标题: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"摘要: {result['snippet'][:100]}...")

            # 如果有重排序分数，显示它
            if "rerank_score" in result:
                print(f"重排序分数: {result['rerank_score']:.4f}")
            if "vector_score" in result:
                print(f"向量相似度分数: {result['vector_score']:.4f}")

    except Exception as e:
        print(f"知识库搜索测试失败: {e}")


def test_direct_milvus_search():
    """直接测试 Milvus 搜索"""
    print("\n=== 直接测试 Milvus 搜索 ===")

    query = "电力系统标准"

    try:
        results = search_milvus(query, 2)

        print(f"查询: {query}")
        print(f"结果数量: {len(results)}")

        for i, result in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(f"标题: {result['title'][:50]}...")
            if "rerank_score" in result:
                print(f"重排序分数: {result['rerank_score']:.4f}")

    except Exception as e:
        print(f"直接 Milvus 搜索测试失败: {e}")


def show_environment_info():
    """显示环境信息"""
    print("=== 环境信息 ===")

    env_vars = [
        "XINFERENCE_HOST",
        "XINFERENCE_PORT",
        "XINFERENCE_EMBEDDING_MODEL",
        "XINFERENCE_RERANK_MODEL",
        "MILVUS_HOST",
        "MILVUS_PORT",
        "MILVUS_COLLECTION",
    ]

    for var in env_vars:
        value = os.getenv(var, "未设置")
        print(f"{var}: {value}")


if __name__ == "__main__":
    print("开始测试 Xinference 集成功能...\n")

    try:
        show_environment_info()
        test_xinference_connection()
        test_xinference_embedding()
        test_milvus_connection()
        test_knowledge_base_search()
        test_direct_milvus_search()

        print("\n=== 测试完成 ===")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()