#!/usr/bin/env python3
"""
测试数据源搜索功能的脚本
"""

import os
import sys

# 添加 src 目录到 Python 路径
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from agent.utils import search_by_data_source, search_milvus, search_web


def test_internet_search():
    """测试互联网搜索"""
    print("=== 测试互联网搜索 ===")
    query = "人工智能发展趋势"
    results = search_by_data_source(query, "internet", 3)

    print(f"查询: {query}")
    print(f"数据源: 互联网")
    print(f"结果数量: {len(results)}")

    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"标题: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"摘要: {result['snippet'][:100]}...")


def test_knowledge_base_search():
    """测试知识库搜索"""
    print("\n=== 测试知识库搜索 ===")
    query = "IEC 标准规范"
    results = search_by_data_source(query, "knowledge_base", 3)

    print(f"查询: {query}")
    print(f"数据源: 私有知识库")
    print(f"结果数量: {len(results)}")

    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"标题: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"摘要: {result['snippet'][:100]}...")


def test_direct_searches():
    """直接测试各个搜索函数"""
    print("\n=== 直接测试搜索函数 ===")

    # 测试网络搜索
    print("\n--- DuckDuckGo 搜索 ---")
    web_results = search_web("Python programming", 2)
    for result in web_results:
        print(f"标题: {result['title'][:50]}...")

    # 测试 Milvus 搜索
    print("\n--- Milvus 搜索 ---")
    milvus_results = search_milvus("技术文档", 2)
    for result in milvus_results:
        print(f"标题: {result['title'][:50]}...")


if __name__ == "__main__":
    print("开始测试数据源搜索功能...\n")

    try:
        test_internet_search()
        test_knowledge_base_search()
        test_direct_searches()

        print("\n=== 测试完成 ===")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()