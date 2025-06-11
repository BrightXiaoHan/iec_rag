"""Milvus 向量数据库搜索实现

这个模块提供了与 Milvus 向量数据库交互的功能，
支持 Xinference 和 OpenAI 兼容的嵌入服务器。
"""

import os
from typing import Dict, List

from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()

# Milvus 配置
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "iec_knowledge_base")

# Xinference 配置
XINFERENCE_HOST = os.getenv("XINFERENCE_HOST", "localhost")
XINFERENCE_PORT = os.getenv("XINFERENCE_PORT", "9997")
XINFERENCE_EMBEDDING_MODEL = os.getenv("XINFERENCE_EMBEDDING_MODEL", "bge-base-en-v1.5")
XINFERENCE_RERANK_MODEL = os.getenv("XINFERENCE_RERANK_MODEL", "bge-reranker-base")

# OpenAI 兼容配置
OPENAI_COMPATIBLE_API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY") or os.getenv(
    "DASHSCOPE_API_KEY"
)
OPENAI_COMPATIBLE_BASE_URL = os.getenv(
    "OPENAI_COMPATIBLE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
OPENAI_COMPATIBLE_EMBEDDING_MODEL = os.getenv(
    "OPENAI_COMPATIBLE_EMBEDDING_MODEL", "text-embedding-v4"
)
OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS = int(
    os.getenv("OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS", "1024")
)

# 嵌入服务提供商选择：xinference, openai_compatible
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "xinference")


def get_openai_compatible_embedding(text: str) -> List[float]:
    """使用 OpenAI 兼容的 API 获取文本嵌入

    Args:
        text: 输入文本

    Returns:
        嵌入向量
    """
    try:
        from openai import OpenAI

        if not OPENAI_COMPATIBLE_API_KEY:
            print("OpenAI 兼容 API Key 未配置")
            return []

        client = OpenAI(
            api_key=OPENAI_COMPATIBLE_API_KEY, base_url=OPENAI_COMPATIBLE_BASE_URL
        )

        # 生成嵌入
        response = client.embeddings.create(
            model=OPENAI_COMPATIBLE_EMBEDDING_MODEL,
            input=[text],
            dimensions=OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS,
            encoding_format="float",
        )

        return response.data[0].embedding

    except Exception as e:
        print(f"OpenAI 兼容嵌入生成失败: {e}")
        return []


def get_xinference_embedding(text: str) -> List[float]:
    """使用 Xinference 获取文本嵌入

    Args:
        text: 输入文本

    Returns:
        嵌入向量
    """
    try:
        from xinference.client import Client

        client = Client(f"http://{XINFERENCE_HOST}:{XINFERENCE_PORT}")

        # 获取嵌入模型
        model = client.get_model(XINFERENCE_EMBEDDING_MODEL)

        # 生成嵌入
        embedding = model.create_embedding(text)

        return embedding["data"][0]["embedding"]

    except Exception as e:
        print(f"Xinference 嵌入生成失败: {e}")
        return []


def get_embedding(text: str) -> List[float]:
    """根据配置选择嵌入服务提供商获取文本嵌入

    Args:
        text: 输入文本

    Returns:
        嵌入向量
    """
    if EMBEDDING_PROVIDER == "openai_compatible":
        return get_openai_compatible_embedding(text)
    else:
        return get_xinference_embedding(text)


def rerank_results_with_xinference(
    query: str, results: List[Dict], top_k: int = 5
) -> List[Dict]:
    """使用 Xinference 重排序搜索结果

    Args:
        query: 查询文本
        results: 搜索结果列表
        top_k: 返回的顶部结果数量

    Returns:
        重排序后的结果
    """
    if not results:
        return results

    try:
        from xinference.client import Client

        client = Client(f"http://{XINFERENCE_HOST}:{XINFERENCE_PORT}")

        # 获取重排序模型
        rerank_model = client.get_model(XINFERENCE_RERANK_MODEL)

        # 准备文档文本
        documents = [result.get("snippet", "") for result in results]

        # 执行重排序
        rerank_response = rerank_model.rerank(
            query=query, documents=documents, top_n=min(top_k, len(documents))
        )

        # 根据重排序结果重新排列
        reranked_results = []
        for item in rerank_response["results"]:
            index = item["index"]
            score = item["relevance_score"]
            if index < len(results):
                result = results[index].copy()
                result["rerank_score"] = score
                reranked_results.append(result)

        return reranked_results

    except Exception as e:
        print(f"Xinference 重排序失败: {e}")
        # 如果重排序失败，返回原始结果
        return results[:top_k]


def search_milvus_real(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """使用 Milvus 向量数据库执行真实的搜索

    Args:
        query: 搜索查询
        num_results: 返回结果数量

    Returns:
        搜索结果列表
    """
    search_results = []

    # 检查是否安装了必要的依赖

    try:
        # 连接到 Milvus
        client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

        # 使用配置的嵌入服务将查询转换为向量
        query_vector = get_embedding(query)

        if not query_vector:
            print("无法生成查询向量")
            return []

        print(f"使用 {EMBEDDING_PROVIDER} 嵌入服务，向量维度: {len(query_vector)}")

        # 搜索参数
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        # 执行搜索，获取更多结果用于重排序
        search_limit = min(num_results * 2, 20)  # 获取2倍结果用于重排序
        results = client.search(
            collection_name=MILVUS_COLLECTION,
            data=[query_vector],
            anns_field="embedding",
            search_params=search_params,
            limit=search_limit,
            output_fields=["title", "content", "url", "source"],
        )

        # 处理搜索结果
        initial_results = []
        for hits in results:
            for hit in hits:
                entity = hit.entity
                initial_results.append(
                    {
                        "title": entity.get("title", "未知标题"),
                        "url": entity.get("url", ""),
                        "snippet": entity.get("content", "")[:300] + "..."
                        if len(entity.get("content", "")) > 300
                        else entity.get("content", ""),
                        "source": entity.get("source", "IEC 知识库"),
                        "vector_score": hit.score,
                    }
                )

        # 如果有 Xinference 重排序模型可用，进行重排序
        if EMBEDDING_PROVIDER == "xinference" or check_xinference_rerank_available():
            search_results = rerank_results_with_xinference(
                query, initial_results, num_results
            )
        else:
            # 如果没有重排序，直接返回向量搜索结果
            search_results = initial_results[:num_results]

    except Exception as e:
        print(f"Milvus 搜索错误: {e}")
        # 如果 Milvus 搜索失败，返回空结果
        search_results = []

    return search_results


def check_milvus_connection() -> bool:
    """检查 Milvus 连接是否可用

    Returns:
        连接状态
    """
    try:
        client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

        # 检查集合是否存在
        has_collection = client.has_collection(MILVUS_COLLECTION)

        return has_collection
    except Exception as e:
        print(f"Milvus 连接检查失败: {e}")
        return False


def check_openai_compatible_connection() -> bool:
    """检查 OpenAI 兼容服务连接是否可用

    Returns:
        连接状态
    """
    try:
        from openai import OpenAI

        if not OPENAI_COMPATIBLE_API_KEY:
            return False

        client = OpenAI(
            api_key=OPENAI_COMPATIBLE_API_KEY, base_url=OPENAI_COMPATIBLE_BASE_URL
        )

        # 尝试生成一个简单的嵌入来测试连接
        response = client.embeddings.create(
            model=OPENAI_COMPATIBLE_EMBEDDING_MODEL,
            input=["test"],
            dimensions=OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS,
            encoding_format="float",
        )

        return len(response.data) > 0

    except Exception as e:
        print(f"OpenAI 兼容服务连接检查失败: {e}")
        return False


def check_xinference_connection() -> bool:
    """检查 Xinference 连接是否可用

    Returns:
        连接状态
    """
    try:
        from xinference.client import Client

        client = Client(f"http://{XINFERENCE_HOST}:{XINFERENCE_PORT}")

        # 尝试列出模型来检查连接
        models = client.list_models()

        # 检查所需的模型是否可用
        embedding_available = any(
            model["model_name"] == XINFERENCE_EMBEDDING_MODEL for model in models
        )
        rerank_available = any(
            model["model_name"] == XINFERENCE_RERANK_MODEL for model in models
        )

        return embedding_available and rerank_available

    except Exception as e:
        print(f"Xinference 连接检查失败: {e}")
        return False


def check_xinference_rerank_available() -> bool:
    """检查 Xinference 重排序模型是否可用

    Returns:
        重排序模型可用状态
    """
    try:
        from xinference.client import Client

        client = Client(f"http://{XINFERENCE_HOST}:{XINFERENCE_PORT}")

        # 尝试列出模型来检查连接
        models = client.list_models()

        # 检查重排序模型是否可用
        rerank_available = any(
            model["model_name"] == XINFERENCE_RERANK_MODEL for model in models
        )

        return rerank_available

    except Exception:
        return False


def check_embedding_service() -> bool:
    """检查配置的嵌入服务是否可用

    Returns:
        嵌入服务可用状态
    """
    if EMBEDDING_PROVIDER == "openai_compatible":
        return check_openai_compatible_connection()
    else:
        return check_xinference_connection()


def initialize_milvus_collection():
    """初始化 Milvus 集合（如果需要的话）
    这个函数可以用于设置集合结构
    """
    try:
        from pymilvus import (
            CollectionSchema,
            DataType,
            FieldSchema,
            MilvusClient,
        )

        client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

        # 根据嵌入服务提供商确定向量维度
        if EMBEDDING_PROVIDER == "openai_compatible":
            embedding_dim = OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS
        else:
            # 大多数 BGE 模型的维度是 768
            embedding_dim = 768

        print(f"使用嵌入维度: {embedding_dim}")

        # 定义集合结构
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(
                name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim
            ),
        ]

        schema = CollectionSchema(fields, "IEC 知识库集合")

        # 创建集合
        client.create_collection(MILVUS_COLLECTION, schema)

        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        client.create_index(MILVUS_COLLECTION, index_params)

        print(f"成功初始化 Milvus 集合: {MILVUS_COLLECTION}")

    except Exception as e:
        print(f"初始化 Milvus 集合失败: {e}")


def launch_xinference_models():
    """启动所需的 Xinference 模型"""
    try:
        from xinference.client import Client

        client = Client(f"http://{XINFERENCE_HOST}:{XINFERENCE_PORT}")

        # 启动嵌入模型
        try:
            client.launch_model(
                model_name=XINFERENCE_EMBEDDING_MODEL, model_type="embedding"
            )
            print(f"成功启动嵌入模型: {XINFERENCE_EMBEDDING_MODEL}")
        except Exception as e:
            print(f"启动嵌入模型失败: {e}")

        # 启动重排序模型
        try:
            client.launch_model(model_name=XINFERENCE_RERANK_MODEL, model_type="rerank")
            print(f"成功启动重排序模型: {XINFERENCE_RERANK_MODEL}")
        except Exception as e:
            print(f"启动重排序模型失败: {e}")

    except Exception as e:
        print(f"启动 Xinference 模型失败: {e}")


def get_embedding_service_info() -> Dict[str, str]:
    """获取当前嵌入服务的信息

    Returns:
        嵌入服务信息
    """
    if EMBEDDING_PROVIDER == "openai_compatible":
        return {
            "provider": "OpenAI Compatible",
            "base_url": OPENAI_COMPATIBLE_BASE_URL,
            "model": OPENAI_COMPATIBLE_EMBEDDING_MODEL,
            "dimensions": str(OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS),
        }
    else:
        return {
            "provider": "Xinference",
            "host": f"{XINFERENCE_HOST}:{XINFERENCE_PORT}",
            "embedding_model": XINFERENCE_EMBEDDING_MODEL,
            "rerank_model": XINFERENCE_RERANK_MODEL,
        }