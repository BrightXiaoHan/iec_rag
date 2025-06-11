# Xinference 集成说明

本项目已经集成了 Xinference 作为嵌入和重排序服务器，替代了 sentence_transformers，提供更强大和灵活的向量搜索能力。

## 功能概述

### 架构变更
- **移除**: sentence_transformers 依赖
- **新增**: xinference-client 集成
- **增强**: 向量搜索 + 重排序的两阶段检索

### 核心组件
1. **Xinference 嵌入服务**: 将文本转换为向量
2. **Milvus 向量搜索**: 基于向量相似度的初步检索
3. **Xinference 重排序服务**: 对搜索结果进行精确重排序

## 安装和配置

### 1. 安装依赖

```bash
# 安装 Python 依赖
pip install xinference-client>=0.9.0 pymilvus>=2.3.0

# 安装 Xinference 服务器
pip install xinference
```

### 2. 环境配置

创建 `.env` 文件并添加以下配置：

```bash
# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=iec_knowledge_base

# Xinference 配置
XINFERENCE_HOST=localhost
XINFERENCE_PORT=9997
XINFERENCE_EMBEDDING_MODEL=bge-base-en-v1.5
XINFERENCE_RERANK_MODEL=bge-reranker-base
```

### 3. 启动 Xinference 服务器

```bash
# 启动 Xinference 服务器
xinference-local --host 0.0.0.0 --port 9997
```

### 4. 启动模型

使用自动化脚本：
```bash
cd backend
python setup_xinference.py
```

或手动启动：
```bash
# 启动嵌入模型
curl -X POST "http://localhost:9997/v1/models" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "bge-base-en-v1.5", "model_type": "embedding"}'

# 启动重排序模型
curl -X POST "http://localhost:9997/v1/models" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "bge-reranker-base", "model_type": "rerank"}'
```

## 使用方式

### 1. 基本搜索流程

```python
from agent.utils import search_by_data_source

# 使用知识库搜索（Xinference + Milvus）
results = search_by_data_source("IEC 61850 标准", "knowledge_base", 5)

# 使用互联网搜索（DuckDuckGo）
results = search_by_data_source("最新技术趋势", "internet", 5)
```

### 2. 两阶段检索过程

当选择"私有知识库"时，系统执行以下步骤：

1. **向量化查询**: 使用 Xinference 将查询文本转换为向量
2. **向量搜索**: 在 Milvus 中执行向量相似度搜索，获取候选结果
3. **重排序**: 使用 Xinference 重排序模型对结果进行精确排序
4. **返回结果**: 返回重排序后的最相关结果

### 3. 前端界面

- 选择"私有知识库"：触发 Xinference + Milvus 搜索
- 选择"互联网"：触发 DuckDuckGo 搜索
- 活动时间线显示相应的搜索类型

## 核心文件说明

### 后端文件

1. **`src/agent/milvus_search.py`**
   - Xinference 客户端集成
   - 嵌入生成和重排序功能
   - Milvus 连接和搜索逻辑

2. **`src/agent/utils.py`**
   - 数据源路由逻辑
   - 智能回退机制

3. **`src/agent/graph.py`**
   - 搜索节点更新
   - 数据源参数传递

4. **`pyproject.toml`**
   - 依赖管理更新

### 工具脚本

1. **`setup_xinference.py`**
   - 自动化模型启动
   - 连接状态检查

2. **`test_xinference_integration.py`**
   - 完整功能测试
   - 连接验证

## 模型配置

### 默认模型

- **嵌入模型**: `bge-base-en-v1.5`
  - 维度: 768
  - 语言: 英文/中文
  - 用途: 文本向量化

- **重排序模型**: `bge-reranker-base`
  - 用途: 查询-文档相关性评分
  - 输出: 相关性分数

### 自定义模型

可以通过环境变量配置不同的模型：

```bash
XINFERENCE_EMBEDDING_MODEL=your-embedding-model
XINFERENCE_RERANK_MODEL=your-rerank-model
```

## 测试和验证

### 1. 运行集成测试

```bash
cd backend
python test_xinference_integration.py
```

### 2. 检查服务状态

```bash
# 检查 Xinference 服务器
curl http://localhost:9997/v1/models

# 检查 Milvus 连接
python -c "from agent.milvus_search import check_milvus_connection; print(check_milvus_connection())"
```

### 3. 测试嵌入功能

```bash
python -c "
from agent.milvus_search import get_xinference_embedding
result = get_xinference_embedding('测试文本')
print(f'嵌入维度: {len(result) if result else 0}')
"
```

## 故障排除

### 常见问题

1. **Xinference 连接失败**
   - 检查服务器是否启动
   - 验证端口配置
   - 确认防火墙设置

2. **模型启动失败**
   - 检查模型名称是否正确
   - 确认系统资源充足
   - 查看 Xinference 日志

3. **Milvus 连接问题**
   - 验证 Milvus 服务状态
   - 检查集合是否存在
   - 确认网络连接

### 日志和调试

系统会自动输出详细的状态信息：

```
使用真实的 Milvus + Xinference 搜索
✅ 嵌入模型测试成功，维度: 768
重排序分数: 0.8542
```

### 回退机制

如果 Xinference 或 Milvus 不可用，系统会：

1. 自动检测连接状态
2. 输出详细错误信息
3. 切换到模拟搜索模式
4. 确保服务不中断

## 性能优化

### 1. 批量处理

对于大量查询，可以使用批量嵌入：

```python
# 批量生成嵌入（需要扩展实现）
embeddings = model.create_embedding([text1, text2, text3])
```

### 2. 缓存策略

考虑为常用查询添加缓存：

```python
# 可以添加 Redis 缓存嵌入结果
cache_key = f"embedding:{hash(text)}"
```

### 3. 模型选择

根据需求选择合适的模型：

- **速度优先**: 使用较小的嵌入模型
- **精度优先**: 使用大型多语言模型
- **中文优化**: 选择中文特化模型

## 扩展功能

### 1. 多模型支持

可以配置多个嵌入模型用于不同场景：

```python
XINFERENCE_EMBEDDING_MODEL_EN=bge-base-en-v1.5
XINFERENCE_EMBEDDING_MODEL_ZH=bge-base-zh-v1.5
```

### 2. 自定义重排序

实现自定义重排序逻辑：

```python
def custom_rerank(query, results):
    # 自定义重排序算法
    pass
```

### 3. 混合搜索

结合多种搜索策略：

```python
def hybrid_search(query):
    # 向量搜索 + 关键词搜索 + 重排序
    pass
```

## 部署建议

### 生产环境

1. **资源配置**
   - CPU: 8+ 核心
   - 内存: 16GB+
   - GPU: 可选，加速推理

2. **服务监控**
   - 健康检查端点
   - 性能指标收集
   - 错误日志监控

3. **高可用性**
   - 多实例部署
   - 负载均衡
   - 故障转移

### Docker 部署

```dockerfile
# Dockerfile 示例
FROM python:3.11-slim

RUN pip install xinference pymilvus

EXPOSE 9997

CMD ["xinference-local", "--host", "0.0.0.0", "--port", "9997"]
```

## 总结

Xinference 集成提供了：

- ✅ 更强大的嵌入能力
- ✅ 精确的重排序功能
- ✅ 灵活的模型配置
- ✅ 智能回退机制
- ✅ 完整的测试覆盖

这个集成显著提升了知识库搜索的准确性和相关性，为用户提供更好的搜索体验。 