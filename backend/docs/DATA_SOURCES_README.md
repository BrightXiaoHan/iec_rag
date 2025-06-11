# 数据源功能说明

本项目已经从原来的 "Effort" 选项改为 "数据源" 选项，支持两种不同的搜索方式：

## 功能概述

### 1. 互联网搜索
- 使用 DuckDuckGo 搜索引擎
- 搜索公开的网络信息
- 适合获取最新的公开信息和一般性知识

### 2. 私有知识库搜索
- 使用 Milvus 向量数据库
- 搜索内部的专业文档和知识
- 适合获取组织内部的专业知识和标准文档

## 前端修改

### 界面变化
- 将原来的 "Effort" 选择器改为 "数据源" 选择器
- 选项从 "Low/Medium/High" 改为 "互联网/私有知识库"
- 活动时间线显示会根据数据源类型显示不同的标题

### 文件修改
- `frontend/src/components/InputForm.tsx`: 修改选择器组件
- `frontend/src/App.tsx`: 更新提交逻辑和活动显示

## 后端修改

### 核心功能
- `backend/src/agent/utils.py`: 添加数据源路由逻辑
- `backend/src/agent/milvus_search.py`: Milvus 搜索实现
- `backend/src/agent/graph.py`: 更新搜索节点逻辑
- `backend/src/agent/state.py`: 添加数据源状态字段

### 搜索逻辑
```python
def search_by_data_source(query: str, data_source: str, num_results: int = 5):
    if data_source == "knowledge_base":
        return search_milvus(query, num_results)
    else:
        return search_web(query, num_results)
```

## Milvus 配置

### 环境变量
创建 `.env` 文件并配置以下变量：

```bash
# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=iec_knowledge_base

# 嵌入模型配置
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 依赖安装
```bash
pip install pymilvus>=2.3.0 sentence-transformers>=2.2.0
```

### 集合初始化
```python
from agent.milvus_search import initialize_milvus_collection
initialize_milvus_collection()
```

## 使用方式

### 1. 启动服务
```bash
# 后端
cd backend
make dev

# 前端
cd frontend
npm run dev
```

### 2. 选择数据源
- 在前端界面选择 "互联网" 或 "私有知识库"
- 输入查询内容
- 点击搜索

### 3. 查看结果
- 活动时间线会显示相应的搜索类型
- 结果会根据选择的数据源返回不同的内容

## 测试

运行测试脚本验证功能：
```bash
cd backend
python test_data_sources.py
```

## 故障排除

### Milvus 连接问题
- 确保 Milvus 服务正在运行
- 检查环境变量配置
- 验证网络连接

### 模拟模式
如果 Milvus 不可用，系统会自动切换到模拟模式，返回示例数据。

## 扩展功能

### 添加新的数据源
1. 在 `utils.py` 中添加新的搜索函数
2. 在 `search_by_data_source` 中添加新的条件分支
3. 在前端添加新的选项

### 自定义 Milvus 集合
修改 `milvus_search.py` 中的集合结构和搜索参数。 