import os
import re
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger
from core.cache import QueryCacheManager

class HybridRetriever:
    def __init__(self, knowledge_base):
        """初始化混合检索器"""
        self.knowledge_base = knowledge_base
        self.vector_store = knowledge_base.get_vector_store()
        self.vector_weight = float(os.getenv("VECTOR_RETRIEVAL_WEIGHT", "0.6"))
        self.keyword_weight = float(os.getenv("KEYWORD_RETRIEVAL_WEIGHT", "0.4"))
        self.top_k = int(os.getenv("TOP_K", "5"))
        
        # 初始化TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            token_pattern=r'\b\w+\b',
            stop_words=None,
            lowercase=True
        )
        
        # 缓存文档内容和TF-IDF矩阵
        self.documents_cache = []
        self.tfidf_matrix = None
        
        # 初始化查询结果缓存管理器
        self.query_cache = QueryCacheManager()
        
        self._build_keyword_index()
    
    def _build_keyword_index(self):
        """构建关键词索引"""
        try:
            # 获取所有文档
            documents = self.vector_store.get()
            if not documents or "documents" not in documents:
                logger.warning("知识库为空，无法构建关键词索引")
                return
            
            # 提取文档内容
            self.documents_cache = []
            for i, doc in enumerate(documents["documents"]):
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                self.documents_cache.append({
                    "content": content,
                    "metadata": metadata,
                    "id": i
                })
            
            # 构建TF-IDF矩阵
            if self.documents_cache:
                texts = [doc["content"] for doc in self.documents_cache]
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                logger.info(f"关键词索引构建完成，共 {len(self.documents_cache)} 个文档")
        except Exception as e:
            logger.error(f"构建关键词索引失败: {str(e)}")
    
    def vector_retrieval(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
        """向量检索"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            # 转换结果格式
            vector_results = []
            for doc, score in results:
                # 计算相似度（1 - 距离）
                similarity = 1.0 - score
                vector_results.append((similarity, {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }))
            
            logger.info(f"向量检索完成，找到 {len(vector_results)} 个结果")
            return vector_results
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
            return []
    
    def keyword_retrieval(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
        """关键词检索"""
        try:
            if not self.documents_cache or self.tfidf_matrix is None:
                return []
            
            # 处理查询
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # 计算相似度
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # 获取top_k结果
            top_indices = similarities.argsort()[-top_k:][::-1]
            keyword_results = []
            
            for idx in top_indices:
                if similarities[idx] > 0:
                    keyword_results.append((float(similarities[idx]), {
                        "content": self.documents_cache[idx]["content"],
                        "metadata": self.documents_cache[idx]["metadata"]
                    }))
            
            logger.info(f"关键词检索完成，找到 {len(keyword_results)} 个结果")
            return keyword_results
        except Exception as e:
            logger.error(f"关键词检索失败: {str(e)}")
            return []
    
    def hybrid_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """混合检索"""
        try:
            # 构建检索参数用于缓存键生成
            retrieval_params = {
                "vector_weight": self.vector_weight,
                "keyword_weight": self.keyword_weight,
                "top_k": top_k
            }
            
            # 尝试从缓存获取结果
            cached_results = self.query_cache.get(query, retrieval_params)
            if cached_results is not None:
                return cached_results
            
            # 执行两种检索
            vector_results = self.vector_retrieval(query, top_k * 2)
            keyword_results = self.keyword_retrieval(query, top_k * 2)
            
            # 融合结果
            combined_results = defaultdict(float)
            result_metadata = {}
            
            # 添加向量检索结果
            for similarity, result in vector_results:
                content_hash = hash(result["content"])
                combined_results[content_hash] += similarity * self.vector_weight
                result_metadata[content_hash] = result
            
            # 添加关键词检索结果
            for similarity, result in keyword_results:
                content_hash = hash(result["content"])
                combined_results[content_hash] += similarity * self.keyword_weight
                if content_hash not in result_metadata:
                    result_metadata[content_hash] = result
            
            # 排序并返回top_k结果
            sorted_results = sorted(
                combined_results.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            final_results = []
            for content_hash, score in sorted_results:
                result = result_metadata[content_hash]
                result["score"] = score
                final_results.append(result)
            
            # 将结果存入缓存
            self.query_cache.set(query, retrieval_params, final_results)
            
            logger.info(f"混合检索完成，返回 {len(final_results)} 个结果")
            return final_results
        except Exception as e:
            logger.error(f"混合检索失败: {str(e)}")
            return []
    
    def get_relevant_documents(self, query: str) -> List[Dict]:
        """获取相关文档"""
        return self.hybrid_retrieval(query, self.top_k)
    
    def update_keyword_index(self):
        """更新关键词索引"""
        self._build_keyword_index()
    
    def adjust_retrieval_weights(self, vector_weight: float, keyword_weight: float):
        """调整检索权重"""
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        logger.info(f"检索权重已调整: 向量={vector_weight}, 关键词={keyword_weight}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self.query_cache.get_stats()
    
    def clear_cache(self):
        """清空查询结果缓存"""
        self.query_cache.clear()
        logger.info("查询结果缓存已清空")
    
    def invalidate_cache_by_query(self, query: str):
        """使特定查询的缓存失效"""
        self.query_cache.invalidate_by_query(query)
        logger.info(f"查询 '{query[:50]}...' 的缓存已失效")
