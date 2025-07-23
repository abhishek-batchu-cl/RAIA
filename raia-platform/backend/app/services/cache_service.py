"""
Redis Cache Service
High-performance caching layer for improved application performance
"""

import asyncio
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import hashlib
import structlog

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

logger = structlog.get_logger(__name__)


class CacheService:
    """
    Redis-based caching service with advanced features
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 3600,
        key_prefix: str = "raia:"
    ):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.redis_client: Optional[aioredis.Redis] = None
        self.connection_pool: Optional[ConnectionPool] = None
        self.logger = logger.bind(service="cache")
        
        # Cache categories for organized management
        self.categories = {
            "model_predictions": {"ttl": 1800, "prefix": "pred:"},      # 30 minutes
            "evaluation_results": {"ttl": 7200, "prefix": "eval:"},     # 2 hours
            "user_sessions": {"ttl": 3600, "prefix": "session:"},       # 1 hour
            "dashboard_data": {"ttl": 900, "prefix": "dashboard:"},     # 15 minutes
            "model_metadata": {"ttl": 14400, "prefix": "model:"},       # 4 hours
            "analytics_data": {"ttl": 1800, "prefix": "analytics:"},    # 30 minutes
            "export_jobs": {"ttl": 86400, "prefix": "export:"},         # 24 hours
            "websocket_data": {"ttl": 300, "prefix": "ws:"},            # 5 minutes
        }
    
    async def initialize(self):
        """Initialize Redis connection pool"""
        try:
            self.connection_pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                decode_responses=False  # We'll handle encoding ourselves
            )
            
            self.redis_client = aioredis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=False
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self.logger.info(
                "Redis cache service initialized",
                redis_url=self.redis_url,
                default_ttl=self.default_ttl
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to initialize Redis connection",
                error=str(e),
                redis_url=self.redis_url
            )
            raise
    
    async def close(self):
        """Close Redis connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()
    
    def _build_key(self, key: str, category: Optional[str] = None) -> str:
        """Build full cache key with prefix and category"""
        if category and category in self.categories:
            prefix = self.categories[category]["prefix"]
            return f"{self.key_prefix}{prefix}{key}"
        return f"{self.key_prefix}{key}"
    
    def _hash_key(self, data: Any) -> str:
        """Generate consistent hash for complex data structures"""
        if isinstance(data, (dict, list)):
            serialized = json.dumps(data, sort_keys=True)
        else:
            serialized = str(data)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        category: Optional[str] = None,
        serialize_method: str = "json"
    ) -> bool:
        """
        Set cache value with optional TTL and category
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            category: Cache category for organization
            serialize_method: Serialization method (json, pickle)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            await self.initialize()
        
        try:
            cache_key = self._build_key(key, category)
            
            # Determine TTL
            if ttl is None:
                if category and category in self.categories:
                    ttl = self.categories[category]["ttl"]
                else:
                    ttl = self.default_ttl
            
            # Serialize value
            if serialize_method == "pickle":
                serialized_value = pickle.dumps(value)
            else:
                serialized_value = json.dumps(value).encode()
            
            # Set with TTL
            result = await self.redis_client.setex(
                cache_key,
                ttl,
                serialized_value
            )
            
            self.logger.debug(
                "Cache set",
                key=cache_key,
                ttl=ttl,
                size_bytes=len(serialized_value),
                category=category
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Cache set failed",
                key=key,
                category=category,
                error=str(e)
            )
            return False
    
    async def get(
        self,
        key: str,
        category: Optional[str] = None,
        serialize_method: str = "json"
    ) -> Optional[Any]:
        """
        Get cache value
        
        Args:
            key: Cache key
            category: Cache category
            serialize_method: Serialization method (json, pickle)
            
        Returns:
            Cached value or None if not found
        """
        if not self.redis_client:
            await self.initialize()
        
        try:
            cache_key = self._build_key(key, category)
            serialized_value = await self.redis_client.get(cache_key)
            
            if serialized_value is None:
                return None
            
            # Deserialize value
            if serialize_method == "pickle":
                value = pickle.loads(serialized_value)
            else:
                value = json.loads(serialized_value.decode())
            
            self.logger.debug(
                "Cache hit",
                key=cache_key,
                category=category
            )
            
            return value
            
        except Exception as e:
            self.logger.error(
                "Cache get failed",
                key=key,
                category=category,
                error=str(e)
            )
            return None
    
    async def delete(self, key: str, category: Optional[str] = None) -> bool:
        """Delete cache entry"""
        if not self.redis_client:
            await self.initialize()
        
        try:
            cache_key = self._build_key(key, category)
            result = await self.redis_client.delete(cache_key)
            
            self.logger.debug(
                "Cache delete",
                key=cache_key,
                category=category,
                deleted=bool(result)
            )
            
            return bool(result)
            
        except Exception as e:
            self.logger.error(
                "Cache delete failed",
                key=key,
                category=category,
                error=str(e)
            )
            return False
    
    async def exists(self, key: str, category: Optional[str] = None) -> bool:
        """Check if cache key exists"""
        if not self.redis_client:
            await self.initialize()
        
        try:
            cache_key = self._build_key(key, category)
            result = await self.redis_client.exists(cache_key)
            return bool(result)
            
        except Exception as e:
            self.logger.error(
                "Cache exists check failed",
                key=key,
                category=category,
                error=str(e)
            )
            return False
    
    async def get_or_set(
        self,
        key: str,
        fetch_function,
        ttl: Optional[int] = None,
        category: Optional[str] = None,
        serialize_method: str = "json"
    ) -> Any:
        """
        Get from cache or fetch and set if not found
        
        Args:
            key: Cache key
            fetch_function: Async function to fetch data if not cached
            ttl: Time to live in seconds
            category: Cache category
            serialize_method: Serialization method
            
        Returns:
            Cached or fetched value
        """
        # Try to get from cache first
        cached_value = await self.get(key, category, serialize_method)
        
        if cached_value is not None:
            return cached_value
        
        # Fetch data
        try:
            if asyncio.iscoroutinefunction(fetch_function):
                fresh_value = await fetch_function()
            else:
                fresh_value = fetch_function()
            
            # Cache the fresh value
            await self.set(key, fresh_value, ttl, category, serialize_method)
            
            return fresh_value
            
        except Exception as e:
            self.logger.error(
                "Fetch function failed",
                key=key,
                error=str(e)
            )
            raise
    
    async def get_pattern(self, pattern: str, category: Optional[str] = None) -> List[Any]:
        """Get all values matching a pattern"""
        if not self.redis_client:
            await self.initialize()
        
        try:
            search_pattern = self._build_key(pattern, category)
            keys = await self.redis_client.keys(search_pattern)
            
            if not keys:
                return []
            
            values = await self.redis_client.mget(keys)
            
            # Deserialize values
            result = []
            for value in values:
                if value:
                    try:
                        deserialized = json.loads(value.decode())
                        result.append(deserialized)
                    except:
                        # Try pickle if JSON fails
                        try:
                            deserialized = pickle.loads(value)
                            result.append(deserialized)
                        except:
                            pass
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Pattern search failed",
                pattern=pattern,
                category=category,
                error=str(e)
            )
            return []
    
    async def clear_category(self, category: str) -> int:
        """Clear all cache entries in a category"""
        if not self.redis_client:
            await self.initialize()
        
        try:
            if category not in self.categories:
                raise ValueError(f"Unknown category: {category}")
            
            pattern = f"{self.key_prefix}{self.categories[category]['prefix']}*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                deleted = await self.redis_client.delete(*keys)
                
                self.logger.info(
                    "Category cleared",
                    category=category,
                    deleted_count=deleted
                )
                
                return deleted
            
            return 0
            
        except Exception as e:
            self.logger.error(
                "Category clear failed",
                category=category,
                error=str(e)
            )
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            await self.initialize()
        
        try:
            info = await self.redis_client.info()
            
            # Get counts by category
            category_counts = {}
            for category, config in self.categories.items():
                pattern = f"{self.key_prefix}{config['prefix']}*"
                keys = await self.redis_client.keys(pattern)
                category_counts[category] = len(keys)
            
            return {
                "redis_info": {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                },
                "category_counts": category_counts,
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
            
        except Exception as e:
            self.logger.error("Failed to get cache stats", error=str(e))
            return {}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage"""
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100
    
    async def flush_all(self) -> bool:
        """Flush all cache entries (use with caution)"""
        if not self.redis_client:
            await self.initialize()
        
        try:
            await self.redis_client.flushdb()
            
            self.logger.warning("All cache entries flushed")
            return True
            
        except Exception as e:
            self.logger.error("Cache flush failed", error=str(e))
            return False
    
    async def expire(self, key: str, ttl: int, category: Optional[str] = None) -> bool:
        """Set expiration time for existing key"""
        if not self.redis_client:
            await self.initialize()
        
        try:
            cache_key = self._build_key(key, category)
            result = await self.redis_client.expire(cache_key, ttl)
            return bool(result)
            
        except Exception as e:
            self.logger.error(
                "Cache expire failed",
                key=key,
                ttl=ttl,
                error=str(e)
            )
            return False
    
    async def get_ttl(self, key: str, category: Optional[str] = None) -> int:
        """Get remaining TTL for a key"""
        if not self.redis_client:
            await self.initialize()
        
        try:
            cache_key = self._build_key(key, category)
            ttl = await self.redis_client.ttl(cache_key)
            return ttl
            
        except Exception as e:
            self.logger.error(
                "TTL check failed",
                key=key,
                error=str(e)
            )
            return -1
    
    async def increment(
        self,
        key: str,
        amount: int = 1,
        category: Optional[str] = None
    ) -> Optional[int]:
        """Increment a numeric value in cache"""
        if not self.redis_client:
            await self.initialize()
        
        try:
            cache_key = self._build_key(key, category)
            result = await self.redis_client.incrby(cache_key, amount)
            return result
            
        except Exception as e:
            self.logger.error(
                "Cache increment failed",
                key=key,
                amount=amount,
                error=str(e)
            )
            return None


# Global cache service instance
cache_service = CacheService()


# Decorator for caching function results
def cached(
    ttl: int = 3600,
    category: str = "default",
    key_builder=None
):
    """
    Decorator to cache function results
    
    Args:
        ttl: Time to live in seconds
        category: Cache category
        key_builder: Function to build cache key from arguments
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key building
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_result = await cache_service.get(cache_key, category)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await cache_service.set(cache_key, result, ttl, category)
            return result
        
        return wrapper
    return decorator