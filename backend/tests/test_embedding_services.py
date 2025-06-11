#!/usr/bin/env python3
"""
测试不同嵌入服务的脚本
"""

import os
import sys

# 添加 src 目录到 Python 路径
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from agent.milvus_search import (
    check_embedding_service,
    check_openai_compatible_connection,
    check_xinference_connection,
    get_embedding,
    get_embedding_service_info,
    get_openai_compatible_embedding,
    get_xinference_embedding,
)
from agent.utils import search_by_data_source


def show_environment_info():
    """显示环境信息"""
    print("=== 环境配置信息 ===")

    env_vars = [
        "EMBEDDING_PROVIDER",
        "OPENAI_COMPATIBLE_API_KEY",
        "DASHSCOPE_API_KEY",
        "OPENAI_COMPATIBLE_BASE_URL",
        "OPENAI_COMPATIBLE_EMBEDDING_MODEL",
        "OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS",
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
        # 隐藏 API Key 的部分内容
        if "API_KEY" in var and value != "未设置":
            value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
        print(f"{var}: {value}")


def test_openai_compatible_embedding():
    """测试 OpenAI 兼容嵌入服务"""
    print("\n=== 测试 OpenAI 兼容嵌入服务 ===")

    # 检查连接
    is_connected = check_openai_compatible_connection()
    print(f"OpenAI 兼容服务连接状态: {'可用' if is_connected else '不可用'}")

    if is_connected:
        # 测试嵌入生成
        test_texts = [
            "风急天高猿啸哀",
            "渚清沙白鸟飞回",
            "无边落木萧萧下",
            "不尽长江滚滚来",
        ]

        for text in test_texts:
            embedding = get_openai_compatible_embedding(text)
            if embedding:
                print(f"文本: {text}")
                print(f"嵌入维度: {len(embedding)}")
                print(f"嵌入向量前3个值: {embedding[:3]}")
                print()
            else:
                print(f"嵌入生成失败: {text}")


def test_xinference_embedding():
    """测试 Xinference 嵌入服务"""
    print("\n=== 测试 Xinference 嵌入服务 ===")

    # 检查连接
    is_connected = check_xinference_connection()
    print(f"Xinference 服务连接状态: {'可用' if is_connected else '不可用'}")

    if is_connected:
        # 测试嵌入生成
        test_text = "这是一个测试文本"
        embedding = get_xinference_embedding(test_text)

        if embedding:
            print(f"文本: {test_text}")
            print(f"嵌入维度: {len(embedding)}")
            print(f"嵌入向量前3个值: {embedding[:3]}")
        else:
            print("嵌入生成失败")


def test_current_embedding_service():
    """测试当前配置的嵌入服务"""
    print("\n=== 测试当前配置的嵌入服务 ===")

    # 显示当前服务信息
    service_info = get_embedding_service_info()
    print("当前嵌入服务配置:")
    for key, value in service_info.items():
        print(f"  {key}: {value}")

    # 检查服务可用性
    is_available = check_embedding_service()
    print(f"\n服务可用性: {'可用' if is_available else '不可用'}")

    if is_available:
        # 测试嵌入生成
        test_text = "IEC 61850 通信协议标准"
        embedding = get_embedding(test_text)

        if embedding:
            print(f"\n测试文本: {test_text}")
            print(f"嵌入维度: {len(embedding)}")
            print(f"嵌入向量前5个值: {embedding[:5]}")
        else:
            print("嵌入生成失败")


def test_knowledge_base_search():
    """测试知识库搜索"""
    print("\n=== 测试知识库搜索 ===")

    query = "电力系统自动化技术"

    try:
        results = search_by_data_source(query, "knowledge_base", 3)

        print(f"查询: {query}")
        print(f"结果数量: {len(results)}")

        for i, result in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(f"标题: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"摘要: {result['snippet'][:100]}...")

            # 显示评分信息
            if "rerank_score" in result:
                print(f"重排序分数: {result['rerank_score']:.4f}")
            if "vector_score" in result:
                print(f"向量相似度分数: {result['vector_score']:.4f}")

    except Exception as e:
        print(f"知识库搜索测试失败: {e}")


def test_embedding_comparison():
    """比较不同嵌入服务的结果"""
    print("\n=== 嵌入服务比较测试 ===")

    test_text = "智能电网技术发展"

    # 测试 OpenAI 兼容服务
    print("OpenAI 兼容服务:")
    openai_embedding = get_openai_compatible_embedding(test_text)
    if openai_embedding:
        print(f"  维度: {len(openai_embedding)}")
        print(f"  前3个值: {openai_embedding[:3]}")
    else:
        print("  不可用")

    # 测试 Xinference 服务
    print("\nXinference 服务:")
    xinference_embedding = get_xinference_embedding(test_text)
    if xinference_embedding:
        print(f"  维度: {len(xinference_embedding)}")
        print(f"  前3个值: {xinference_embedding[:3]}")
    else:
        print("  不可用")


def test_batch_embedding():
    """测试批量嵌入（如果支持）"""
    print("\n=== 批量嵌入测试 ===")

    # 这里可以扩展实现批量嵌入功能
    print("批量嵌入功能待实现...")


if __name__ == "__main__":
    print("开始测试嵌入服务功能...\n")

    try:
        show_environment_info()
        test_current_embedding_service()
        test_openai_compatible_embedding()
        test_xinference_embedding()
        test_embedding_comparison()
        test_knowledge_base_search()
        test_batch_embedding()

        print("\n=== 测试完成 ===")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()