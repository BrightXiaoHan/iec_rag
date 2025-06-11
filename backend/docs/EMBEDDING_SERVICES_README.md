# 嵌入服务集成说明

本项目现在支持多种嵌入服务，包括 Xinference 和 OpenAI 兼容的服务（如阿里云百炼、智谱 AI 等）。

## 支持的嵌入服务

### 1. Xinference（本地部署）
- **优势**: 完全本地化，数据隐私性好，支持多种开源模型
- **劣势**: 需要本地部署和维护，资源消耗较大
- **适用场景**: 对数据隐私要求高，有本地计算资源

### 2. OpenAI 兼容服务（云端 API）
- **优势**: 无需本地部署，调用简单，性能稳定
- **劣势**: 需要网络连接，可能有数据隐私考虑
- **适用场景**: 快速开发，不想维护本地服务

#### 支持的 OpenAI 兼容服务提供商：
- **阿里云百炼**: `text-embedding-v4` (1024维)
- **智谱 AI**: `embedding-2` (1024维)
- **月之暗面 Kimi**: `moonshot-v1-8k` (1536维)
- **OpenAI**: `text-embedding-3-small/large`

## 配置方法

### 环境变量配置

复制 `embedding_services_env_example.txt` 的内容到你的 `.env` 文件中：

```bash
# 选择嵌入服务提供商
EMBEDDING_PROVIDER=openai_compatible  # 或 xinference

# OpenAI 兼容服务配置（以阿里云百炼为例）
DASHSCOPE_API_KEY=your_api_key_here
OPENAI_COMPATIBLE_API_KEY=your_api_key_here
OPENAI_COMPATIBLE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_COMPATIBLE_EMBEDDING_MODEL=text-embedding-v4
OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS=1024

# Xinference 配置
XINFERENCE_HOST=localhost
XINFERENCE_PORT=9997
XINFERENCE_EMBEDDING_MODEL=bge-base-en-v1.5
XINFERENCE_RERANK_MODEL=bge-reranker-base

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=iec_knowledge_base
```

### 不同服务商的具体配置

#### 阿里云百炼
```bash
EMBEDDING_PROVIDER=openai_compatible
DASHSCOPE_API_KEY=your_dashscope_api_key
OPENAI_COMPATIBLE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_COMPATIBLE_EMBEDDING_MODEL=text-embedding-v4
OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS=1024
```

#### 智谱 AI
```bash
EMBEDDING_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_API_KEY=your_zhipu_api_key
OPENAI_COMPATIBLE_BASE_URL=https://open.bigmodel.cn/api/paas/v4
OPENAI_COMPATIBLE_EMBEDDING_MODEL=embedding-2
OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS=1024
```

#### 月之暗面 Kimi
```bash
EMBEDDING_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_API_KEY=your_kimi_api_key
OPENAI_COMPATIBLE_BASE_URL=https://api.moonshot.cn/v1
OPENAI_COMPATIBLE_EMBEDDING_MODEL=moonshot-v1-8k
OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS=1536
```

## 使用方法

### 1. 测试嵌入服务

```bash
# 测试所有嵌入服务功能
python test_embedding_services.py

# 演示 OpenAI 兼容服务（需要设置 API Key）
python demo_openai_embedding.py
```

### 2. 在代码中使用

```python
from agent.milvus_search import get_embedding, check_embedding_service

# 检查服务可用性
if check_embedding_service():
    # 生成嵌入
    embedding = get_embedding("你的文本")
    print(f"嵌入维度: {len(embedding)}")
else:
    print("嵌入服务不可用")
```

### 3. 知识库搜索

```python
from agent.utils import search_by_data_source

# 使用私有知识库搜索
results = search_by_data_source("查询内容", "knowledge_base", 5)
for result in results:
    print(f"标题: {result['title']}")
    print(f"摘要: {result['snippet']}")
```

## 功能特性

### 1. 智能回退机制
- 如果配置的嵌入服务不可用，自动使用模拟数据
- 保证系统的稳定性和可用性

### 2. 两阶段检索（仅 Xinference）
- **第一阶段**: 向量相似度搜索
- **第二阶段**: 重排序优化结果
- 提供更准确的搜索结果

### 3. 多维度支持
- 支持不同维度的嵌入向量
- 自动适配 Milvus 集合结构

### 4. 服务状态监控
- 实时检查服务连接状态
- 提供详细的错误信息和诊断

## 性能对比

| 服务类型 | 延迟 | 准确性 | 成本 | 隐私性 |
|---------|------|--------|------|--------|
| Xinference | 中等 | 高 | 低（硬件成本） | 高 |
| 阿里云百炼 | 低 | 高 | 中等 | 中等 |
| 智谱 AI | 低 | 高 | 中等 | 中等 |
| OpenAI | 低 | 很高 | 高 | 低 |

## 故障排除

### 常见问题

1. **"OpenAI 兼容 API Key 未配置"**
   - 检查环境变量是否正确设置
   - 确认 API Key 有效性

2. **"Xinference 连接检查失败"**
   - 确认 Xinference 服务器已启动
   - 检查主机和端口配置

3. **"Milvus 连接不可用"**
   - 确认 Milvus 服务器运行状态
   - 检查集合是否存在

4. **"嵌入维度不匹配"**
   - 确认 Milvus 集合的向量维度设置
   - 重新初始化集合或调整嵌入维度

### 调试步骤

1. 运行测试脚本查看详细错误信息
2. 检查环境变量配置
3. 验证服务连接状态
4. 查看日志输出

## 最佳实践

1. **生产环境建议**:
   - 使用云端 API 服务保证稳定性
   - 设置合理的超时和重试机制
   - 监控 API 调用量和成本

2. **开发环境建议**:
   - 使用 Xinference 进行本地开发
   - 配置模拟数据作为后备方案

3. **数据隐私考虑**:
   - 敏感数据使用本地 Xinference
   - 公开数据可使用云端服务

## 扩展开发

如需添加新的嵌入服务提供商：

1. 在 `milvus_search.py` 中添加新的嵌入函数
2. 更新 `get_embedding()` 函数的路由逻辑
3. 添加相应的连接检查函数
4. 更新环境变量配置

## 技术支持

如遇到问题，请：
1. 查看本文档的故障排除部分
2. 运行测试脚本获取详细信息
3. 检查相关服务的官方文档 