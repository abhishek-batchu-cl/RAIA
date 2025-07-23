"""
Cache Management API Endpoints
Redis cache administration and monitoring
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Form
from typing import Dict, List, Optional, Any

from app.core.auth import get_current_active_user, require_admin_role
from app.models.schemas import User
from app.services.cache_service import cache_service

router = APIRouter()


@router.get("/stats")
async def get_cache_stats(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get comprehensive cache statistics
    
    Returns:
        Cache performance metrics and usage statistics
    """
    try:
        stats = await cache_service.get_stats()
        
        return {
            "status": "success",
            "data": {
                "redis_stats": stats.get("redis_info", {}),
                "category_breakdown": stats.get("category_counts", {}),
                "cache_hit_rate": stats.get("hit_rate", 0.0),
                "available_categories": list(cache_service.categories.keys()),
                "timestamp": "2024-01-30T15:30:00Z"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.post("/clear-category")
async def clear_cache_category(
    category: str = Form(...),
    current_user: User = Depends(require_admin_role)
) -> Dict[str, Any]:
    """
    Clear all cache entries in a specific category (Admin only)
    
    Args:
        category: Cache category to clear
        current_user: Authenticated admin user
        
    Returns:
        Number of entries cleared
    """
    try:
        if category not in cache_service.categories:
            available_categories = list(cache_service.categories.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid category. Available categories: {available_categories}"
            )
        
        deleted_count = await cache_service.clear_category(category)
        
        return {
            "status": "success",
            "message": f"Cache category '{category}' cleared successfully",
            "data": {
                "category": category,
                "entries_deleted": deleted_count,
                "timestamp": "2024-01-30T15:30:00Z"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache category: {str(e)}")


@router.get("/key/{key}")
async def get_cache_key(
    key: str,
    category: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get specific cache key value
    
    Args:
        key: Cache key to retrieve
        category: Cache category (optional)
        current_user: Authenticated user
        
    Returns:
        Cache key value and metadata
    """
    try:
        # Check if key exists
        exists = await cache_service.exists(key, category)
        if not exists:
            raise HTTPException(status_code=404, detail="Cache key not found")
        
        # Get value and TTL
        value = await cache_service.get(key, category)
        ttl = await cache_service.get_ttl(key, category)
        
        return {
            "status": "success",
            "data": {
                "key": key,
                "category": category,
                "value": value,
                "ttl_seconds": ttl,
                "exists": True,
                "retrieved_at": "2024-01-30T15:30:00Z"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache key: {str(e)}")


@router.delete("/key/{key}")
async def delete_cache_key(
    key: str,
    category: Optional[str] = Query(None),
    current_user: User = Depends(require_admin_role)
) -> Dict[str, Any]:
    """
    Delete specific cache key (Admin only)
    
    Args:
        key: Cache key to delete
        category: Cache category (optional)
        current_user: Authenticated admin user
        
    Returns:
        Deletion confirmation
    """
    try:
        deleted = await cache_service.delete(key, category)
        
        return {
            "status": "success",
            "message": f"Cache key deleted: {deleted}",
            "data": {
                "key": key,
                "category": category,
                "deleted": deleted,
                "timestamp": "2024-01-30T15:30:00Z"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete cache key: {str(e)}")


@router.post("/key/{key}/expire")
async def set_cache_expiration(
    key: str,
    ttl_seconds: int = Form(..., ge=1),
    category: Optional[str] = Form(None),
    current_user: User = Depends(require_admin_role)
) -> Dict[str, Any]:
    """
    Set expiration time for cache key (Admin only)
    
    Args:
        key: Cache key
        ttl_seconds: Time to live in seconds
        category: Cache category (optional)
        current_user: Authenticated admin user
        
    Returns:
        Expiration set confirmation
    """
    try:
        success = await cache_service.expire(key, ttl_seconds, category)
        
        if not success:
            raise HTTPException(status_code=404, detail="Cache key not found")
        
        return {
            "status": "success",
            "message": "Cache key expiration updated",
            "data": {
                "key": key,
                "category": category,
                "ttl_seconds": ttl_seconds,
                "updated": success,
                "timestamp": "2024-01-30T15:30:00Z"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set cache expiration: {str(e)}")


@router.get("/categories")
async def get_cache_categories(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get available cache categories and their configurations
    
    Returns:
        Cache categories with TTL and usage information
    """
    try:
        stats = await cache_service.get_stats()
        category_counts = stats.get("category_counts", {})
        
        categories_info = []
        for category, config in cache_service.categories.items():
            categories_info.append({
                "name": category,
                "prefix": config["prefix"],
                "default_ttl": config["ttl"],
                "current_entries": category_counts.get(category, 0),
                "description": f"Cache category for {category.replace('_', ' ')}"
            })
        
        return {
            "status": "success",
            "data": {
                "categories": categories_info,
                "total_categories": len(categories_info),
                "timestamp": "2024-01-30T15:30:00Z"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache categories: {str(e)}")


@router.post("/flush")
async def flush_all_cache(
    confirm: bool = Form(False),
    current_user: User = Depends(require_admin_role)
) -> Dict[str, Any]:
    """
    Flush all cache entries (Admin only - use with extreme caution)
    
    Args:
        confirm: Confirmation flag (must be True)
        current_user: Authenticated admin user
        
    Returns:
        Flush operation result
    """
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="Confirmation required. Set confirm=True to proceed."
        )
    
    try:
        success = await cache_service.flush_all()
        
        return {
            "status": "success",
            "message": "All cache entries flushed successfully",
            "data": {
                "flushed": success,
                "timestamp": "2024-01-30T15:30:00Z",
                "performed_by": current_user.email
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to flush cache: {str(e)}")


@router.get("/search")
async def search_cache_keys(
    pattern: str = Query(...),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Search cache keys by pattern
    
    Args:
        pattern: Search pattern (supports wildcards)
        category: Cache category to search in
        limit: Maximum number of results
        current_user: Authenticated user
        
    Returns:
        Matching cache keys and their metadata
    """
    try:
        # Add wildcard if not present
        if "*" not in pattern:
            pattern = f"*{pattern}*"
        
        values = await cache_service.get_pattern(pattern, category)
        
        # Limit results
        limited_values = values[:limit]
        
        return {
            "status": "success",
            "data": {
                "pattern": pattern,
                "category": category,
                "results": limited_values,
                "result_count": len(limited_values),
                "total_found": len(values),
                "limited": len(values) > limit,
                "timestamp": "2024-01-30T15:30:00Z"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache search failed: {str(e)}")


@router.get("/health")
async def cache_health_check(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Check cache service health
    
    Returns:
        Cache service health status and performance metrics
    """
    try:
        # Test basic cache operations
        test_key = f"health_check_{current_user.id}"
        test_value = {"timestamp": "2024-01-30T15:30:00Z", "test": True}
        
        # Test set operation
        set_success = await cache_service.set(test_key, test_value, ttl=60)
        
        # Test get operation
        retrieved_value = await cache_service.get(test_key)
        
        # Test delete operation
        delete_success = await cache_service.delete(test_key)
        
        # Get stats for additional health info
        stats = await cache_service.get_stats()
        
        health_status = "healthy" if (set_success and retrieved_value and delete_success) else "unhealthy"
        
        return {
            "status": health_status,
            "data": {
                "cache_operations": {
                    "set": set_success,
                    "get": retrieved_value is not None,
                    "delete": delete_success
                },
                "redis_connection": "connected",
                "memory_usage": stats.get("redis_info", {}).get("used_memory_human", "unknown"),
                "hit_rate": stats.get("hit_rate", 0.0),
                "timestamp": "2024-01-30T15:30:00Z"
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-30T15:30:00Z"
        }


@router.get("/performance")
async def get_cache_performance_metrics(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get detailed cache performance metrics
    
    Returns:
        Performance metrics including hit rates, memory usage, and operation counts
    """
    try:
        stats = await cache_service.get_stats()
        redis_info = stats.get("redis_info", {})
        
        return {
            "status": "success",
            "data": {
                "hit_rate": stats.get("hit_rate", 0.0),
                "memory_usage": {
                    "used_memory": redis_info.get("used_memory", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "memory_efficiency": "good" if redis_info.get("used_memory", 0) < 1073741824 else "monitor"  # 1GB threshold
                },
                "connection_info": {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "total_commands_processed": redis_info.get("total_commands_processed", 0)
                },
                "cache_effectiveness": {
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0),
                    "hit_rate_category": "excellent" if stats.get("hit_rate", 0) > 80 else 
                                      "good" if stats.get("hit_rate", 0) > 60 else "needs_improvement"
                },
                "category_distribution": stats.get("category_counts", {}),
                "timestamp": "2024-01-30T15:30:00Z"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")