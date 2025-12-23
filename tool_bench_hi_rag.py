import os
import sys
import json
import pickle
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# LangChain imports
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import TOOL_BENCH_DIR


@dataclass
class RetrievalConfig:
    """检索配置"""
    # Embedding 配置
    embedding_api_key: str
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "embedding"
    
    # Rerank 配置
    rerank_api_key: Optional[str] = None
    rerank_base_url: str = "https://api.openai.com/v1"
    rerank_model: str = "rerank"
    
    # 检索参数
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    top_k: int = 10
    rerank_top_k: int = 5
    
    # RRF 参数
    rrf_k: int = 60  # RRF 平滑参数，常用值为 60
    
    # 持久化路径
    index_dir: str = "./faiss_index"
    
    # 性能优化参数
    enable_cache: bool = True
    batch_size: int = 32
    
    def __post_init__(self):
        """初始化后处理"""
        if self.rerank_api_key is None:
            self.rerank_api_key = self.embedding_api_key


class ReciprocalRankFusion:
    """
    实现 Reciprocal Rank Fusion (RRF) 算法
    
    RRF 是一种无监督的融合方法，用于合并来自不同检索系统的结果。
    公式: RRF(d) = Σ (w_i / (k + rank_i(d)))
    其中 k 是平滑参数（通常设为60），rank_i(d) 是文档 d 在第 i 个列表中的排名，w_i 是权重
    """
    
    def __init__(self, k: int = 60):
        """
        初始化 RRF
        
        Args:
            k: 平滑参数，用于避免首位文档的得分过高，通常设为 60
        """
        self.k = k
    
    def fuse(
        self, 
        ranked_lists: List[List[Document]], 
        weights: Optional[List[float]] = None
    ) -> List[Document]:
        """
        融合多个排序列表
        
        Args:
            ranked_lists: 多个排序后的文档列表
            weights: 每个列表的权重，如果为 None 则平均分配
        
        Returns:
            融合后的文档列表（按 RRF 分数排序）
        """
        if not ranked_lists:
            return []
        
        # 如果没有提供权重，使用平均权重
        if weights is None:
            weights = [1.0] * len(ranked_lists)
        else:
            # 归一化权重
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights] if total_weight > 0 else weights
        
        # 计算每个文档的 RRF 分数
        doc_scores = defaultdict(float)
        doc_map = {}  # 存储文档对象
        
        for list_idx, doc_list in enumerate(ranked_lists):
            for rank, doc in enumerate(doc_list):
                # 使用文档的唯一标识作为 key
                doc_key = self._get_doc_key(doc)
                
                # RRF 分数计算: weight / (k + rank + 1)
                # rank + 1 是因为排名从 1 开始，而不是 0
                rrf_score = weights[list_idx] / (self.k + rank + 1)
                doc_scores[doc_key] += rrf_score
                
                # 保存文档对象（如果还没有保存）
                if doc_key not in doc_map:
                    doc_map[doc_key] = doc
        
        # 按 RRF 分数排序
        sorted_doc_keys = sorted(
            doc_scores.keys(), 
            key=lambda k: doc_scores[k], 
            reverse=True
        )
        
        # 构建结果列表，并附加 RRF 分数到 metadata
        result_docs = []
        for doc_key in sorted_doc_keys:
            doc = doc_map[doc_key]
            # 创建新的 Document 对象，添加 RRF 分数
            new_metadata = doc.metadata.copy()
            new_metadata['rrf_score'] = doc_scores[doc_key]
            result_docs.append(
                Document(page_content=doc.page_content, metadata=new_metadata)
            )
        
        return result_docs
    
    @staticmethod
    def _get_doc_key(doc: Document) -> str:
        """
        获取文档的唯一标识
        
        优先使用 metadata 中的 ID，否则使用内容的哈希
        """
        if 'id' in doc.metadata:
            return str(doc.metadata['id'])
        return str(hash(doc.page_content))


class HybridRetrievalSystem:
    """混合检索系统，支持 BM25、向量检索和 RRF 融合"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.documents: List[Document] = []
        self.bm25_retriever = None
        self.vectorstore = None
        self.vector_retriever = None
        self.rrf_fusion = ReciprocalRankFusion(k=config.rrf_k)
        
        # 初始化 Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url
        )
        
        # 创建索引目录
        Path(config.index_dir).mkdir(parents=True, exist_ok=True)
        
        # 查询缓存
        self._query_cache = {} if config.enable_cache else None
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        添加文档到检索系统
        
        Args:
            texts: 文档文本列表
            metadatas: 文档元数据列表
        """
        print(f" 添加 {len(texts)} 个文档到系统...")
        
        # 创建 Document 对象
        if metadatas is None:
            metadatas = [{"id": i} for i in range(len(texts))]
        
        self.documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        
        # 初始化 BM25 检索器
        print("  初始化 BM25 检索器...")
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = self.config.top_k
        
        # 初始化向量检索器
        print("  初始化向量检索器 (使用 OpenAI Embeddings)...")
        self.vectorstore = FAISS.from_documents(
            self.documents,
            self.embeddings
        )
        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )
        
        print(f" 文档添加完成!\n")
    
    def save_index(self, index_name: str = "default"):
        """
        保存索引到磁盘
        
        Args:
            index_name: 索引名称
        """
        print(f"保存索引: {index_name}")
        
        index_path = Path(self.config.index_dir) / index_name
        index_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 FAISS 向量索引
        if self.vectorstore:
            faiss_path = str(index_path / "faiss_index")
            self.vectorstore.save_local(faiss_path)
            print(f"  FAISS 索引已保存")
        
        # 保存 BM25 索引（保存原始文档）
        if self.documents:
            docs_path = index_path / "documents.pkl"
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            print(f"  文档已保存: {len(self.documents)} 个")
        
        # 保存配置
        config_path = index_path / "config.json"
        config_dict = {
            "embedding_model": self.config.embedding_model,
            "top_k": self.config.top_k,
            "bm25_weight": self.config.bm25_weight,
            "vector_weight": self.config.vector_weight,
            "rrf_k": self.config.rrf_k
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f" 索引保存完成: {index_path}\n")
    
    def load_index(self, index_name: str = "default"):
        """
        从磁盘加载索引
        
        Args:
            index_name: 索引名称
        """
        print(f" 加载索引: {index_name}")
        
        index_path = Path(self.config.index_dir) / index_name
        
        if not index_path.exists():
            raise FileNotFoundError(f"索引不存在: {index_path}")
        
        # 加载 FAISS 向量索引
        faiss_path = str(index_path / "faiss_index")
        if Path(faiss_path).exists():
            self.vectorstore = FAISS.load_local(
                faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.vector_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.top_k}
            )
            print(f"  FAISS 索引已加载")
        
        # 加载文档并重建 BM25 索引
        docs_path = index_path / "documents.pkl"
        if docs_path.exists():
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            # 重建 BM25 索引
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = self.config.top_k
            print(f"  文档已加载，BM25 索引已重建")
        
        print(f" 索引加载完成! (共 {len(self.documents)} 个文档)\n")
    
    def retrieve_bm25(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        BM25 检索
        
        Args:
            query: 查询文本
            k: 返回文档数量
        """
        if k is not None:
            original_k = self.bm25_retriever.k
            self.bm25_retriever.k = k
            results = self.bm25_retriever.invoke(query)
            self.bm25_retriever.k = original_k
        else:
            results = self.bm25_retriever.invoke(query)
        return results
    
    def retrieve_vector(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        向量检索
        
        Args:
            query: 查询文本
            k: 返回文档数量
        """
        if k is not None:
            results = self.vectorstore.similarity_search(query, k=k)
        else:
            results = self.vector_retriever.invoke(query)
        return results
    
    def retrieve_hybrid_rrf(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        混合检索 (使用 RRF 融合)
        
        Args:
            query: 查询文本
            k: 返回的文档数量
        """
        # 检查缓存
        cache_key = f"hybrid_rrf_{query}_{k}"
        if self._query_cache is not None and cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        # 获取 BM25 和向量检索结果
        search_k = k or self.config.top_k
        bm25_results = self.retrieve_bm25(query, k=search_k)
        vector_results = self.retrieve_vector(query, k=search_k)
        
        # 使用 RRF 融合
        fused_results = self.rrf_fusion.fuse(
            [bm25_results, vector_results],
            weights=[self.config.bm25_weight, self.config.vector_weight]
        )
        
        # 返回 top-k 结果
        result = fused_results[:search_k]
        
        # 缓存结果
        if self._query_cache is not None:
            self._query_cache[cache_key] = result
        
        return result
    
    def rerank_with_model(self, query: str, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        使用 Rerank 模型进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
        """
        if not documents:
            return []
        
        print(f"使用 Rerank 模型 ({self.config.rerank_model}) 进行重排序...")
        
        doc_texts = [doc.page_content for doc in documents]
        
        try:
            rerank_results = self._call_rerank_api(query, doc_texts)
            
            # 按相关性分数排序
            sorted_results = sorted(
                rerank_results,
                key=lambda x: x['relevance_score'],
                reverse=True
            )[:self.config.rerank_top_k]
            
            # 构建返回结果
            reranked_docs = []
            for rank, result in enumerate(sorted_results):
                idx = result['index']
                reranked_docs.append({
                    "rank": rank + 1,
                    "content": documents[idx].page_content,
                    "metadata": documents[idx].metadata,
                    "relevance_score": result['relevance_score'],
                    "original_index": idx
                })
            
            print(f"  重排序完成，返回 Top {len(reranked_docs)} 结果\n")
            return reranked_docs
            
        except Exception as e:
            print(f" Rerank 失败: {e}")
            print(f"  使用原始排序返回 Top {self.config.rerank_top_k} 结果\n")
            # 降级方案：返回原始排序
            return [
                {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": doc.metadata.get('rrf_score', 0.0),
                    "original_index": i
                }
                for i, doc in enumerate(documents[:self.config.rerank_top_k])
            ]
    
    def _call_rerank_api(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """
        调用 Rerank API
        
        Args:
            query: 查询文本
            documents: 文档文本列表
        """
        url = f"{self.config.rerank_base_url.rstrip('/')}/rerank"
        
        headers = {
            "Authorization": f"Bearer {self.config.rerank_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": self.config.rerank_top_k,
            "return_documents": False
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"Rerank API 错误: {response.status_code} - {response.text}")
        
        result = response.json()
        
        if "results" in result:
            return [
                {
                    "index": item["index"],
                    "relevance_score": item.get("relevance_score", 0.0)
                }
                for item in result["results"]
            ]
        else:
            raise Exception(f"未知的 Rerank API 响应格式: {result}")
    
    def search(self, query: str, method: str = "hybrid", k: Optional[int] = None) -> Dict[str, Any]:
        """
        执行检索
        
        Args:
            query: 查询文本
            method: 检索方法 ("bm25", "vector", "hybrid", "hybrid_rerank")
            k: 返回的文档数量（可选）
        
        Returns:
            检索结果字典
        """
        results = {
            "query": query,
            "method": method,
            "results": []
        }
        
        if method == "bm25":
            docs = self.retrieve_bm25(query, k=k)
            results["results"] = [
                {
                    "rank": i + 1, 
                    "content": doc.page_content, 
                    "metadata": doc.metadata
                }
                for i, doc in enumerate(docs)
            ]
        
        elif method == "vector":
            docs = self.retrieve_vector(query, k=k)
            results["results"] = [
                {
                    "rank": i + 1, 
                    "content": doc.page_content, 
                    "metadata": doc.metadata
                }
                for i, doc in enumerate(docs)
            ]
        
        elif method == "hybrid":
            docs = self.retrieve_hybrid_rrf(query, k=k)
            results["results"] = [
                {
                    "rank": i + 1, 
                    "content": doc.page_content, 
                    "metadata": doc.metadata,
                    "rrf_score": doc.metadata.get('rrf_score', 0.0)
                }
                for i, doc in enumerate(docs)
            ]
        
        elif method == "hybrid_rerank":
            docs = self.retrieve_hybrid_rrf(query, k=k or self.config.top_k)
            results["results"] = self.rerank_with_model(query, docs)
        
        else:
            raise ValueError(f"未知的检索方法: {method}")
        
        return results
    
    def clear_cache(self):
        """清除查询缓存"""
        if self._query_cache is not None:
            self._query_cache.clear()
            print(" 缓存已清除")


class MultiStageRetrievalSystem:
    """
    多级检索系统
    
    支持多级检索流程：
    1. 第一级：使用粗粒度索引（type_service）进行初步筛选
    2. 第二级：使用细粒度索引（type_service_tool）进行精确重排序
    """
    
    def __init__(self, config: RetrievalConfig, index_dir: str = "./faiss_index"):
        """
        初始化多级检索系统
        
        Args:
            config: 检索配置
            index_dir: 索引目录
        """
        self.config = config
        self.index_dir = Path(index_dir)
        self.systems: Dict[str, HybridRetrievalSystem] = {}
    
    def load_index(self, index_name: str, system_key: Optional[str] = None):
        """
        加载索引
        
        Args:
            index_name: 索引名称
            system_key: 系统键名（如果为 None，使用 index_name）
        """
        key = system_key or index_name
        system = HybridRetrievalSystem(self.config)
        system.load_index(index_name)
        self.systems[key] = system
        return system
    
    def multi_stage_search(
        self,
        query: str,
        stage1_index: str = "type_service_index",
        stage2_index: str = "type_service_tool_index",
        stage1_top_k: int = 10,
        stage2_top_k: int = 5
    ) -> Dict[str, Any]:
        """
        多级检索：先用粗粒度索引筛选，再用细粒度索引重排序
        
        Args:
            query: 查询文本
            stage1_index: 第一级索引名称（粗粒度）
            stage2_index: 第二级索引名称（细粒度）
            stage1_top_k: 第一级返回的文档数量
            stage2_top_k: 第二级返回的文档数量（最终结果）
        
        Returns:
            检索结果字典
        """
        # 确保索引已加载
        if stage1_index not in self.systems:
            self.load_index(stage1_index)
        
        if stage2_index not in self.systems:
            self.load_index(stage2_index)
        
        # 第一级：粗粒度检索（BM25 + Vector）
        stage1_system = self.systems[stage1_index]
        stage1_docs = stage1_system.retrieve_hybrid_rrf(query, k=stage1_top_k)
        
        # 提取第一级结果的类型和服务信息，用于构建第二级查询上下文
        stage1_types_services = set()
        for doc in stage1_docs:
            content = doc.page_content
            stage1_types_services.add(content)
        
        # 第二级：在细粒度索引中进行检索和重排序
        stage2_system = self.systems[stage2_index]
        
        # 方案1: 直接使用查询在细粒度索引中检索
        stage2_docs = stage2_system.retrieve_hybrid_rrf(query, k=stage1_top_k * 2)
        
        # 过滤：只保留与第一级结果相关的文档
        filtered_stage2_docs = []
        for doc in stage2_docs:
            content = doc.page_content
            # 检查是否包含第一级的 type_service
            for ts in stage1_types_services:
                if ts in content:
                    filtered_stage2_docs.append(doc)
                    break
        
        # 使用 Rerank 模型进行精确重排序
        if filtered_stage2_docs:
            final_results = stage2_system.rerank_with_model(query, filtered_stage2_docs[:stage1_top_k])
        else:
            # 降级方案：如果过滤后没有结果，直接在第二级索引重排序
            stage2_docs_direct = stage2_system.retrieve_hybrid_rrf(query, k=stage1_top_k)
            final_results = stage2_system.rerank_with_model(query, stage2_docs_direct)
        
        # 构建结果
        result = {
            "query": query,
            "method": "multi_stage",
            "stage1_index": stage1_index,
            "stage2_index": stage2_index,
            "stage1_top_k": stage1_top_k,
            "stage2_top_k": stage2_top_k,
            "stage1_results_count": len(stage1_docs),
            "stage2_filtered_count": len(filtered_stage2_docs),
            "results": final_results[:stage2_top_k]
        }
        
        return result


def print_results(results: Dict[str, Any], max_content_length: int = 100):
    """
    打印检索结果
    
    Args:
        results: 检索结果字典
        max_content_length: 内容显示的最大长度
    """
    print("\n" + "="*80)
    print(f"检索结果 - {results['method'].upper()}")
    print("="*80)
    print(f"Query: {results['query']}\n")
    
    # 打印多级检索的额外信息
    if results['method'] == 'multi_stage':
        print(f"第一级索引: {results['stage1_index']}")
        print(f"第二级索引: {results['stage2_index']}")
        print(f"第一级筛选: {results['stage1_results_count']} 个候选")
        print(f"第二级过滤: {results['stage2_filtered_count']} 个相关文档")
        print()
    
    for item in results['results']:
        print(f"Rank {item['rank']}:")
        content = item['content']
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        print(f"  Content: {content}")
        
        if 'metadata' in item:
            meta_display = {k: v for k, v in item['metadata'].items() 
                          if k not in ['rrf_score', 'id']}
            if meta_display:
                print(f"  Metadata: {meta_display}")
        
        if 'rrf_score' in item and item['rrf_score'] is not None:
            print(f"  RRF Score: {item['rrf_score']:.6f}")
        
        if 'relevance_score' in item and item['relevance_score'] is not None:
            print(f"  Relevance Score: {item['relevance_score']:.4f}")
        
        print()


def build_indexes(config: RetrievalConfig, limit: Optional[int] = None):
    """
    构建所有索引
    
    Args:
        config: 检索配置
        limit: 限制处理的文档数量（用于测试）
    """
    print("="*80)
    print("="*80 + "\n")
    
    # 加载数据
    print(f"加载数据: {TOOL_BENCH_DIR}")
    documents_data = json.loads(open(TOOL_BENCH_DIR).read())
    
    if limit:
        documents_data = documents_data[:limit]
        print(f"  限制处理前 {limit} 个文档")
    
    print(f"✓ 加载完成，共 {len(documents_data)} 个文档\n")
    
    # 1. 构建 type_service 索引
    print("构建索引 1/3: type_service_index")
    print("-" * 80)
    texts_type_service = [
        f"type: {doc['type']} service: {doc['service']}" 
        for doc in documents_data
    ]
    metadatas_type_service = [
        {
            "id": i, 
            "type": documents_data[i]['type'],
            "service": documents_data[i]['service']
        } 
        for i in range(len(documents_data))
    ]
    
    system_type_service = HybridRetrievalSystem(config)
    system_type_service.add_documents(texts_type_service, metadatas_type_service)
    system_type_service.save_index("type_service_index")
    
    # 2. 构建 type_service_tool 索引
    print("构建索引 2/3: type_service_tool_index")
    print("-" * 80)
    texts_type_service_tool = [
        f"type: {doc['type']} service: {doc['service']} tool: {doc['tool']}" 
        for doc in documents_data
    ]
    metadatas_type_service_tool = [
        {
            "id": i, 
            "type": documents_data[i]['type'],
            "service": documents_data[i]['service'],
            "tool": documents_data[i]['tool']
        } 
        for i in range(len(documents_data))
    ]
    
    system_type_service_tool = HybridRetrievalSystem(config)
    system_type_service_tool.add_documents(texts_type_service_tool, metadatas_type_service_tool)
    system_type_service_tool.save_index("type_service_tool_index")
    
    # 3. 构建 tool 索引
    print("构建索引 3/3: tool_index")
    print("-" * 80)
    texts_tool = [f"tool: {doc['tool']}" for doc in documents_data]
    metadatas_tool = [
        {
            "id": i, 
            "tool": documents_data[i]['tool']
        } 
        for i in range(len(documents_data))
    ]
    
    system_tool = HybridRetrievalSystem(config)
    system_tool.add_documents(texts_tool, metadatas_tool)
    system_tool.save_index("tool_index")
    
    print("="*80)
    print(" 所有索引构建完成!")
    print("="*80 + "\n")


def calculate_dcg(relevance_scores: List[float], k: int) -> float:
    """
    计算 Discounted Cumulative Gain (DCG)
    
    Args:
        relevance_scores: 相关性分数列表（按排名顺序）
        k: 计算前k个结果
    
    Returns:
        DCG值
    """
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        # DCG公式: rel / log2(i+2)，i从0开始，所以位置是i+1，log的底数位置是i+2
        dcg += rel / np.log2(i + 2)
    return dcg


def calculate_ndcg(predicted_services: List[str], relevant_services: List[str], k: int) -> float:
    """
    计算 Normalized Discounted Cumulative Gain (NDCG@k)
    
    Args:
        predicted_services: 预测的服务列表（按排名顺序）
        relevant_services: 相关的服务列表（ground truth）
        k: 计算前k个结果
    
    Returns:
        NDCG@k值
    """
    # 构建相关性分数列表：如果预测的service在relevant_services中，则为1，否则为0
    relevance_scores = [
        1.0 if service in relevant_services else 0.0 
        for service in predicted_services[:k]
    ]
    
    # 计算DCG
    dcg = calculate_dcg(relevance_scores, k)
    
    # 计算理想DCG (IDCG)：假设前k个结果都是相关的
    ideal_relevance = [1.0] * min(k, len(relevant_services)) + [0.0] * max(0, k - len(relevant_services))
    idcg = calculate_dcg(ideal_relevance, k)
    
    # 计算NDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(
    query_list: List[str], 
    label_list: List[List[str]], 
    multi_stage_system: MultiStageRetrievalSystem,
    verbose: bool = True
) -> Dict[str, float]:
    """
    评测检索系统性能
    
    Args:
        query_list: 查询列表
        label_list: 每个查询对应的相关服务列表
        multi_stage_system: 多级检索系统
        verbose: 是否打印详细信息
    
    Returns:
        评测指标字典
    """
    ndcg_at_1_list = []
    ndcg_at_3_list = []
    ndcg_at_5_list = []
    
    if verbose:
        print("\n" + "="*80)
        print("开始评测")
        print("="*80 + "\n")
    
    for idx, (query, relevant_services) in enumerate(zip(query_list, label_list)):
        if verbose:
            print(f"处理查询 {idx+1}/{len(query_list)}: {query[:80]}...")
        
        # 执行检索
        results = multi_stage_system.multi_stage_search(
            query=query,
            stage1_index="type_service_index",
            stage2_index="type_service_tool_index",
            stage1_top_k=10,
            stage2_top_k=5
        )
        
        # 提取预测的services
        predicted_services = [
            result['metadata']['service'] 
            for result in results['results']
        ]
        
        # 计算NDCG指标
        ndcg_1 = calculate_ndcg(predicted_services, relevant_services, k=1)
        ndcg_3 = calculate_ndcg(predicted_services, relevant_services, k=3)
        ndcg_5 = calculate_ndcg(predicted_services, relevant_services, k=5)
        
        ndcg_at_1_list.append(ndcg_1)
        ndcg_at_3_list.append(ndcg_3)
        ndcg_at_5_list.append(ndcg_5)
        
        if verbose:
            print(f"  NDCG@1: {ndcg_1:.4f}, NDCG@3: {ndcg_3:.4f}, NDCG@5: {ndcg_5:.4f}")
            print(f"  预测服务 (Top 3): {predicted_services[:3]}")
            print(f"  相关服务: {relevant_services[:3] if len(relevant_services) > 3 else relevant_services}")
            print()
    
    # 计算平均指标
    avg_ndcg_1 = np.mean(ndcg_at_1_list)
    avg_ndcg_3 = np.mean(ndcg_at_3_list)
    avg_ndcg_5 = np.mean(ndcg_at_5_list)
    
    print("="*80)
    print("评测结果汇总")
    print("="*80)
    print(f"平均 NDCG@1: {avg_ndcg_1:.4f}")
    print(f"平均 NDCG@3: {avg_ndcg_3:.4f}")
    print(f"平均 NDCG@5: {avg_ndcg_5:.4f}")
    print(f"总查询数: {len(query_list)}")
    print("="*80 + "\n")
    
    return {
        "ndcg@1": float(avg_ndcg_1),
        "ndcg@3": float(avg_ndcg_3),
        "ndcg@5": float(avg_ndcg_5),
        "num_queries": len(query_list),
        "detailed_ndcg@1": [float(x) for x in ndcg_at_1_list],
        "detailed_ndcg@3": [float(x) for x in ndcg_at_3_list],
        "detailed_ndcg@5": [float(x) for x in ndcg_at_5_list]
    }


def main():
    """主函数 - 演示系统使用和评测"""
    
    # 配置
    config = RetrievalConfig(
        # Embedding 配置
        embedding_api_key="EMPTY",
        embedding_base_url="http://172.20.98.51:8083/v1",
        embedding_model="qwen3_embedding",
        
        # Rerank 配置
        rerank_api_key="EMPTY",
        rerank_base_url="http://172.20.98.51:8085/v1",
        rerank_model="bge_rerank",
        
        # 检索参数
        bm25_weight=0.5,
        vector_weight=0.5,
        top_k=10,
        rerank_top_k=5,
        rrf_k=60,  # RRF 平滑参数
        
        # 持久化路径
        index_dir="./faiss_index",
        
        # 性能优化
        enable_cache=True
    )
    
    # 构建索引（仅第一次运行时需要，如果已经构建可以注释掉）
    build_indexes(config, limit=None)  
    
    # 加载查询和标签
    tool_bench_base_dir = os.path.dirname(TOOL_BENCH_DIR)
    query_path = [os.path.join(tool_bench_base_dir, f'G{i}_query.json') for i in range(1, 4)]
    query_list = []
    label_list = []

    for path in query_path:
        if os.path.exists(path):
            queries = json.loads(open(path).read())
            query_list.append([query_i['query'] for query_i in queries])
            label_list.append([query_i['relevant APIs'] for query_i in queries])
        else:
            print(f"警告: 文件不存在 {path}")
            query_list.append([])
            label_list.append([])
    
    # 选择测试集
    if len(sys.argv) > 1:
        action = sys.argv[1]
        try:
            test_idx = int(action) - 1
            if 0 <= test_idx < 3:
                querys = query_list[test_idx]
                labels = label_list[test_idx]
                print(f"\n选择测试集: G{test_idx+1} ({len(querys)} 个查询)\n")
            else:
                print("错误: 请输入 1、2 或 3")
                return
        except ValueError:
            print("错误: 请输入有效的数字 (1、2 或 3)")
            return
    else:
        print("使用方法:")
        print("  python retrieval.py 1  # Tool Bench G1 测试")
        print("  python retrieval.py 2  # Tool Bench G2 测试")
        print("  python retrieval.py 3  # Tool Bench G3 测试")
        return
    
    # 初始化多级检索系统
    multi_stage_system = MultiStageRetrievalSystem(config)
    
    # 执行评测
    evaluation_results = evaluate_retrieval(querys, labels, multi_stage_system, verbose=True)
    
    # 保存评测结果
    output_path = f"./evaluation_results_G{test_idx+1}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"评测结果已保存到: {output_path}\n")


if __name__ == "__main__":
    main()