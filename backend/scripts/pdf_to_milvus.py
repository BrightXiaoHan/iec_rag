#!/usr/bin/env python3
"""PDF文档解析并录入Milvus数据库脚本

这个脚本使用LangChain解析backend/papers文件夹中的PDF文档，
并将解析后的文档内容录入到Milvus向量数据库中。

使用方法:
    python pdf_to_milvus.py [--batch-size 10] [--chunk-size 1000] [--chunk-overlap 200]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# LangChain imports
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PyMilvus imports
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

# 导入现有的嵌入服务配置
from src.agent.milvus_search import (
    EMBEDDING_PROVIDER,
    MILVUS_COLLECTION,
    MILVUS_TOKEN,
    MILVUS_URI,
    OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS,
    check_embedding_service,
    check_milvus_connection,
    get_embedding,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_to_milvus.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 常量配置
PAPERS_DIR = project_root / "papers"
DEFAULT_CHUNK_SIZE = 32000
DEFAULT_CHUNK_OVERLAP = 1024
DEFAULT_BATCH_SIZE = 10


def setup_milvus_collection(client: MilvusClient, force_recreate: bool = False) -> bool:
    """设置Milvus集合
    
    Args:
        client: Milvus客户端
        force_recreate: 是否强制重新创建集合
        
    Returns:
        是否成功设置集合
    """
    collection_exists = client.has_collection(MILVUS_COLLECTION)
    
    # 检查现有集合是否需要重新创建
    need_recreate = force_recreate
    if collection_exists and not force_recreate:
        try:
            collection_info = client.describe_collection(MILVUS_COLLECTION)
            # 检查是否有vector和text字段（简化schema的基本字段）
            field_names = [field["name"] for field in collection_info.get("fields", [])]
            if "vector" not in field_names:
                logger.info("检测到集合schema不匹配，需要重新创建")
                need_recreate = True
        except Exception as e:
            logger.warning(f"检查集合schema失败: {e}")
    
    if collection_exists and need_recreate:
        logger.info(f"删除现有集合: {MILVUS_COLLECTION}")
        client.drop_collection(MILVUS_COLLECTION)
        collection_exists = False
    
    if not collection_exists:
        logger.info(f"创建新集合: {MILVUS_COLLECTION}")
        
        # 根据嵌入服务提供商确定向量维度
        if EMBEDDING_PROVIDER == "openai_compatible":
            embedding_dim = OPENAI_COMPATIBLE_EMBEDDING_DIMENSIONS
        else:
            # BGE模型通常是768维
            embedding_dim = 768
        
        logger.info(f"使用嵌入维度: {embedding_dim}")
        
        # 使用 MilvusClient 的简化集合创建方式
        # 这会创建一个包含 id, vector 字段的基本集合
        client.create_collection(
            collection_name=MILVUS_COLLECTION,
            dimension=embedding_dim,
            metric_type="L2",
            index_type="IVF_FLAT",
            auto_id=True
        )
        
        logger.info(f"成功创建集合和索引: {MILVUS_COLLECTION}")
    
    # 检查集合的schema
    try:
        collection_info = client.describe_collection(MILVUS_COLLECTION)
        logger.info(f"集合schema: {collection_info}")
        
        # 显示字段信息
        if "fields" in collection_info:
            for field in collection_info["fields"]:
                logger.info(f"  字段: {field.get('name', 'unknown')} - 类型: {field.get('type', 'unknown')}")
    except Exception as e:
        logger.warning(f"无法获取集合schema信息: {e}")
    
    return True


def load_and_split_pdf(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """加载并分割PDF文档
    
    Args:
        pdf_path: PDF文件路径
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小
        
    Returns:
        分割后的文档列表
    """
    logger.info(f"加载PDF文档: {pdf_path}")
    
    # 使用PyPDFLoader加载PDF
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    
    logger.info(f"PDF包含 {len(pages)} 页")
    
    # 使用RecursiveCharacterTextSplitter分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 分割所有页面
    all_splits = text_splitter.split_documents(pages)
    
    # 为每个分割添加额外的元数据
    for i, split in enumerate(all_splits):
        split.metadata.update({
            "file_path": str(pdf_path),
            "source": pdf_path.stem,
            "chunk_index": i
        })
    
    logger.info(f"文档分割为 {len(all_splits)} 个文本块")
    return all_splits


def generate_embeddings_batch(documents: List[Document]) -> List[List[float]]:
    """批量生成文档嵌入
    
    Args:
        documents: 文档列表
        
    Returns:
        嵌入向量列表
    """
    embeddings = []
    
    for doc in tqdm(documents, desc="生成嵌入向量"):
        embedding = get_embedding(doc.page_content)
        if not embedding:
            logger.warning(f"无法为文档生成嵌入: {doc.metadata.get('source', 'unknown')}")
            # 如果无法生成嵌入，跳过这个文档
            continue
        
        # 验证嵌入向量格式
        if not isinstance(embedding, list):
            logger.error(f"嵌入向量类型错误: {type(embedding)}, 应该是list")
            continue
            
        # 确保所有元素都是float类型
        try:
            embedding_floats = [float(x) for x in embedding]
            embeddings.append(embedding_floats)
            logger.debug(f"生成嵌入向量维度: {len(embedding_floats)}")
        except (ValueError, TypeError) as e:
            logger.error(f"嵌入向量转换为float失败: {e}")
            continue
    
    return embeddings


def prepare_milvus_data(documents: List[Document], embeddings: List[List[float]]) -> List[Dict[str, Any]]:
    """准备插入Milvus的数据
    
    Args:
        documents: 文档列表
        embeddings: 嵌入向量列表
        
    Returns:
        准备好的数据列表
    """
    if len(documents) != len(embeddings):
        logger.error(f"文档数量({len(documents)})与嵌入向量数量({len(embeddings)})不匹配")
        return []
    
    data = []
    for doc, embedding in zip(documents, embeddings):
        # 提取页码信息
        page_number = doc.metadata.get("page", 0)
        if isinstance(page_number, str):
            page_number = int(page_number) if page_number.isdigit() else 0
        
        # 生成标题（使用文件名和页码）
        source_name = doc.metadata.get("source", "unknown")
        title = f"{source_name} - Page {page_number}" if page_number > 0 else source_name
        
        # 组合所有信息到一个文本字段中，因为简化schema可能只有基本字段
        combined_text = f"Title: {title}\n"
        combined_text += f"Source: {source_name}\n"
        combined_text += f"Page: {page_number}\n"
        combined_text += f"File: {doc.metadata.get('file_path', '')}\n"
        combined_text += f"Content: {doc.page_content}"
        
        # 对于MilvusClient的简化schema，通常只需要vector字段
        # 其他信息可以存储在text字段中
        data.append({
            "vector": embedding,
            "text": combined_text[:8000]  # 限制长度
        })
    
    return data


def insert_to_milvus(client: MilvusClient, data: List[Dict[str, Any]], batch_size: int) -> bool:
    """批量插入数据到Milvus
    
    Args:
        client: Milvus客户端
        data: 要插入的数据
        batch_size: 批次大小
        
    Returns:
        是否成功插入
    """
    if not data:
        logger.warning("没有数据需要插入")
        return True
    
    logger.info(f"开始插入 {len(data)} 条记录到Milvus，批次大小: {batch_size}")
    
    total_inserted = 0
    
    # 分批插入
    for i in tqdm(range(0, len(data), batch_size), desc="插入数据"):
        batch = data[i:i + batch_size]
        
        # 准备批次数据 - 使用简化的schema格式
        batch_data = []
        for item in batch:
            # 确保vector是List[float]格式
            vector = item["vector"]
            if not isinstance(vector, list):
                logger.error(f"嵌入向量格式错误: {type(vector)}")
                continue
                
            entity_dict = {
                "vector": vector,
                "text": item["text"]
            }
            batch_data.append(entity_dict)
        
        if not batch_data:
            logger.warning("批次中没有有效数据")
            continue
            
        # 插入数据
        result = client.insert(MILVUS_COLLECTION, batch_data)
        total_inserted += len(batch_data)
        
        logger.info(f"批次 {i//batch_size + 1}: 插入 {len(batch_data)} 条记录")
    
    logger.info(f"总共插入 {total_inserted} 条记录")
    return True


def process_single_pdf(pdf_path: Path, client: MilvusClient, chunk_size: int, 
                      chunk_overlap: int, batch_size: int) -> bool:
    """处理单个PDF文件
    
    Args:
        pdf_path: PDF文件路径
        client: Milvus客户端
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠
        batch_size: 批次大小
        
    Returns:
        是否成功处理
    """
    logger.info(f"开始处理PDF文件: {pdf_path}")
    
    # 加载并分割PDF
    documents = load_and_split_pdf(pdf_path, chunk_size, chunk_overlap)
    if not documents:
        logger.warning(f"PDF文件没有内容: {pdf_path}")
        return False
    
    # 生成嵌入向量
    embeddings = generate_embeddings_batch(documents)
    if not embeddings:
        logger.error(f"无法为PDF生成嵌入向量: {pdf_path}")
        return False
    
    # 如果嵌入数量少于文档数量，需要过滤文档
    if len(embeddings) < len(documents):
        logger.warning(f"部分文档无法生成嵌入，过滤后剩余 {len(embeddings)} 个文档")
        # 重新生成文档列表，只保留成功生成嵌入的文档
        valid_documents = []
        embedding_index = 0
        for doc in documents:
            embedding = get_embedding(doc.page_content)
            if embedding:
                valid_documents.append(doc)
                embedding_index += 1
        documents = valid_documents
    
    # 准备Milvus数据
    milvus_data = prepare_milvus_data(documents, embeddings)
    if not milvus_data:
        logger.error(f"无法准备Milvus数据: {pdf_path}")
        return False
    
    # 插入到Milvus
    success = insert_to_milvus(client, milvus_data, batch_size)
    
    if success:
        logger.info(f"成功处理PDF文件: {pdf_path}")
    else:
        logger.error(f"处理PDF文件失败: {pdf_path}")
    
    return success


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="解析PDF文档并录入Milvus数据库")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                       help=f"批次大小 (默认: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                       help=f"文本块大小 (默认: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                       help=f"文本块重叠大小 (默认: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--force-recreate", action="store_true",
                       help="强制重新创建Milvus集合")
    parser.add_argument("--pdf-file", type=str,
                       help="指定单个PDF文件进行处理（相对于papers目录）")
    
    args = parser.parse_args()
    
    logger.info("开始PDF文档解析和录入流程")
    logger.info(f"配置参数: batch_size={args.batch_size}, chunk_size={args.chunk_size}, "
               f"chunk_overlap={args.chunk_overlap}")
    
    # 检查papers目录
    if not PAPERS_DIR.exists():
        logger.error(f"Papers目录不存在: {PAPERS_DIR}")
        return 1
    
    # 检查嵌入服务
    if not check_embedding_service():
        logger.error("嵌入服务不可用，请检查配置")
        return 1
    
    # 检查Milvus连接
    if not check_milvus_connection():
        logger.error("Milvus连接不可用，请检查配置")
        return 1
    
    # 连接到Milvus
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    
    # 设置集合
    if not setup_milvus_collection(client, args.force_recreate):
        logger.error("设置Milvus集合失败")
        return 1
    
    # 获取要处理的PDF文件列表
    if args.pdf_file:
        # 处理指定的单个文件
        pdf_path = PAPERS_DIR / args.pdf_file
        if not pdf_path.exists():
            logger.error(f"指定的PDF文件不存在: {pdf_path}")
            return 1
        pdf_files = [pdf_path]
    else:
        # 处理所有PDF文件
        pdf_files = list(PAPERS_DIR.glob("*.pdf"))
        if not pdf_files:
            logger.error(f"在{PAPERS_DIR}中没有找到PDF文件")
            return 1
    
    logger.info(f"找到 {len(pdf_files)} 个PDF文件需要处理")
    
    # 处理每个PDF文件
    success_count = 0
    for pdf_path in pdf_files:
        if process_single_pdf(pdf_path, client, args.chunk_size, 
                             args.chunk_overlap, args.batch_size):
            success_count += 1
        else:
            logger.error(f"处理失败: {pdf_path}")
    
    # 加载集合到内存
    logger.info("加载集合到内存...")
    client.load_collection(MILVUS_COLLECTION)
    
    # 统计结果
    logger.info(f"处理完成: 成功 {success_count}/{len(pdf_files)} 个文件")
    
    # 检查集合中的数据量
    collection_info = client.get_collection_stats(MILVUS_COLLECTION)
    logger.info(f"集合 {MILVUS_COLLECTION} 中共有 {collection_info['row_count']} 条记录")
    
    return 0 if success_count == len(pdf_files) else 1


if __name__ == "__main__":
    exit(main()) 