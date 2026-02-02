import os
import hashlib
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
from collections import OrderedDict


class QueryCacheManager:
    """查询结果缓存管理器"""
    
    def __init__(self):
        """初始化缓存管理器"""
        self.cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))  # 缓存过期时间（秒），默认1小时
        self.cache_max_size = int(os.getenv("CACHE_MAX_SIZE", "1000"))  # 最大缓存条目数
        
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_queries": 0
        }
        
        logger.info(f"查询缓存管理器初始化完成 - 启用: {self.cache_enabled}, TTL: {self.cache_ttl}秒, 最大容量: {self.cache_max_size}")
    
    def _generate_cache_key(self, query: str, retrieval_params: Dict[str, Any]) -> str:
        """生成缓存键"""
        key_data = f"{query}_{retrieval_params}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """检查缓存是否过期"""
        if self.cache_ttl <= 0:
            return False
        
        created_time = cache_entry.get("created_time", 0)
        return time.time() - created_time > self.cache_ttl
    
    def _cleanup_expired(self):
        """清理过期缓存"""
        if not self.cache_enabled:
            return
        
        expired_keys = []
        for key, entry in self.cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期缓存条目")
    
    def _evict_if_needed(self):
        """如果缓存已满，淘汰最旧的条目"""
        if not self.cache_enabled:
            return
        
        while len(self.cache) >= self.cache_max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"缓存已满，淘汰最旧的缓存条目: {oldest_key}")
    
    def get(self, query: str, retrieval_params: Dict[str, Any]) -> Optional[List[Dict]]:
        """从缓存获取查询结果"""
        if not self.cache_enabled:
            return None
        
        self.cache_stats["total_queries"] += 1
        
        cache_key = self._generate_cache_key(query, retrieval_params)
        
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            
            if self._is_expired(cache_entry):
                del self.cache[cache_key]
                self.cache_stats["misses"] += 1
                logger.debug(f"缓存已过期: {query[:50]}...")
                return None
            
            self.cache_stats["hits"] += 1
            self.cache.move_to_end(cache_key)  # 更新访问顺序
            logger.debug(f"缓存命中: {query[:50]}...")
            return cache_entry["results"]
        
        self.cache_stats["misses"] += 1
        logger.debug(f"缓存未命中: {query[:50]}...")
        return None
    
    def set(self, query: str, retrieval_params: Dict[str, Any], results: List[Dict]):
        """设置查询结果到缓存"""
        if not self.cache_enabled:
            return
        
        cache_key = self._generate_cache_key(query, retrieval_params)
        
        self._cleanup_expired()
        self._evict_if_needed()
        
        self.cache[cache_key] = {
            "query": query,
            "params": retrieval_params,
            "results": results,
            "created_time": time.time()
        }
        
        logger.debug(f"缓存已设置: {query[:50]}...")
    
    def clear(self):
        """清空所有缓存"""
        self.cache.clear()
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_queries": 0
        }
        logger.info("所有缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        hit_rate = 0
        if self.cache_stats["total_queries"] > 0:
            hit_rate = (self.cache_stats["hits"] / self.cache_stats["total_queries"]) * 100
        
        return {
            "enabled": self.cache_enabled,
            "total_entries": len(self.cache),
            "max_size": self.cache_max_size,
            "ttl_seconds": self.cache_ttl,
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "total_queries": self.cache_stats["total_queries"],
            "hit_rate": f"{hit_rate:.2f}%"
        }
    
    def invalidate_by_query(self, query: str):
        """使特定查询的缓存失效"""
        if not self.cache_enabled:
            return
        
        keys_to_remove = []
        for key, entry in self.cache.items():
            if entry["query"] == query:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
        
        if keys_to_remove:
            logger.info(f"使 {len(keys_to_remove)} 个缓存条目失效: {query[:50]}...")
    
    def invalidate_all(self):
        """使所有缓存失效（等同于清空）"""
        self.clear()
