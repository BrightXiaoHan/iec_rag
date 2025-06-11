# PDF文档解析和录入Milvus数据库脚本

这个脚本使用LangChain解析`backend/papers`文件夹中的PDF文档，并将解析后的文档内容录入到Milvus向量数据库中。

## 功能特性

- 🔍 使用LangChain的PyPDFLoader加载PDF文档
- ✂️ 使用RecursiveCharacterTextSplitter智能分割文档
- 🧠 支持多种嵌入服务（Xinference、OpenAI兼容API）
- 📊 批量处理和插入数据到Milvus
- 📝 详细的日志记录和进度显示
- 🔧 灵活的配置选项

## 安装依赖

```bash
# 进入scripts目录
cd backend/scripts

# 安装依赖
pip install -r requirements.txt
```

## 配置要求

在运行脚本之前，请确保以下服务已正确配置：

### 1. Milvus数据库
- 确保Milvus服务正在运行
- 配置环境变量：`MILVUS_URI`、`MILVUS_TOKEN`、`MILVUS_COLLECTION`

### 2. 嵌入服务
选择以下其中一种嵌入服务：

#### Xinference (默认)
```bash
export EMBEDDING_PROVIDER=xinference
export XINFERENCE_HOST=localhost
export XINFERENCE_PORT=9997
export XINFERENCE_EMBEDDING_MODEL=bge-base-en-v1.5
```

#### OpenAI兼容API
```bash
export EMBEDDING_PROVIDER=openai_compatible
export OPENAI_COMPATIBLE_API_KEY=your_api_key
export OPENAI_COMPATIBLE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENAI_COMPATIBLE_EMBEDDING_MODEL=text-embedding-v4
export OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS=1024
```

## 使用方法

### 基本用法

```bash
# 处理所有PDF文件
python pdf_to_milvus.py

# 处理指定的PDF文件
python pdf_to_milvus.py --pdf-file "IEC 62933-1-2024.pdf"

# 强制重新创建Milvus集合
python pdf_to_milvus.py --force-recreate
```

### 高级选项

```bash
# 自定义参数
python pdf_to_milvus.py \
    --batch-size 20 \
    --chunk-size 1500 \
    --chunk-overlap 300 \
    --force-recreate
```

### 参数说明

- `--batch-size`: 批次大小，控制每次插入Milvus的记录数量（默认：10）
- `--chunk-size`: 文本块大小，控制文档分割的块大小（默认：1000）
- `--chunk-overlap`: 文本块重叠大小，控制相邻块之间的重叠（默认：200）
- `--force-recreate`: 强制重新创建Milvus集合（会删除现有数据）
- `--pdf-file`: 指定单个PDF文件进行处理（相对于papers目录）

## 工作流程

1. **检查环境**: 验证papers目录、嵌入服务和Milvus连接
2. **设置集合**: 创建或使用现有的Milvus集合
3. **加载PDF**: 使用PyPDFLoader逐页加载PDF文档
4. **文档分割**: 使用RecursiveCharacterTextSplitter智能分割文档
5. **生成嵌入**: 为每个文档块生成向量嵌入
6. **批量插入**: 将数据批量插入到Milvus数据库
7. **加载集合**: 将集合加载到内存以供搜索

## 数据结构

插入到Milvus的每条记录包含以下字段：

- `id`: 自动生成的主键
- `title`: 文档标题（文件名 + 页码）
- `content`: 文档内容（最大5000字符）
- `url`: URL（PDF文档为空）
- `source`: 源文件名
- `page_number`: 页码
- `file_path`: 文件路径
- `embedding`: 向量嵌入

## 日志和监控

脚本会生成详细的日志：
- 控制台输出：实时显示处理进度
- 日志文件：`pdf_to_milvus.log`记录详细信息

## 故障排除

### 常见问题

1. **嵌入服务连接失败**
   - 检查嵌入服务是否正在运行
   - 验证API密钥和配置

2. **Milvus连接失败**
   - 确认Milvus服务状态
   - 检查连接配置

3. **PDF解析失败**
   - 确认PDF文件完整性
   - 检查文件权限

4. **内存不足**
   - 减小batch_size参数
   - 减小chunk_size参数

### 性能优化

- 增加`batch_size`可以提高插入速度，但会占用更多内存
- 调整`chunk_size`和`chunk_overlap`可以优化文档分割质量
- 使用SSD存储可以提高PDF读取速度

## 示例输出

```
2024-01-01 10:00:00 - INFO - 开始PDF文档解析和录入流程
2024-01-01 10:00:00 - INFO - 配置参数: batch_size=10, chunk_size=1000, chunk_overlap=200
2024-01-01 10:00:01 - INFO - 找到 15 个PDF文件需要处理
2024-01-01 10:00:01 - INFO - 加载PDF文档: /path/to/IEC 62933-1-2024.pdf
2024-01-01 10:00:02 - INFO - PDF包含 42 页
2024-01-01 10:00:03 - INFO - 文档分割为 156 个文本块
生成嵌入向量: 100%|██████████| 156/156 [00:30<00:00,  5.20it/s]
插入数据: 100%|██████████| 16/16 [00:05<00:00,  3.20it/s]
2024-01-01 10:00:38 - INFO - 成功处理PDF文件: /path/to/IEC 62933-1-2024.pdf
2024-01-01 10:15:42 - INFO - 处理完成: 成功 15/15 个文件
2024-01-01 10:15:43 - INFO - 集合 iec_knowledge_base 中共有 2340 条记录
```

## 注意事项

- 首次运行时会创建Milvus集合，后续运行会使用现有集合
- 大型PDF文件可能需要较长处理时间
- 确保有足够的磁盘空间存储向量数据
- 建议在处理大量文档前先测试单个文件 