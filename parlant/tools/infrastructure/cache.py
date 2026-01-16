"""
Caching layer for frequently accessed data.

This module provides in-memory caching to reduce API calls and improve performance:
- Policy document caching
- Ticket data caching (short TTL)
- Configuration caching
- LRU eviction policy
"""

import time
from typing import Any, Optional, Dict, Tuple
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class CacheEntry:
    """Single cache entry with TTL."""
    
    def __init__(self, value: Any, ttl_seconds: float):
        """
        Initialize cache entry.
        
        Args:
            value: The cached value
            ttl_seconds: Time-to-live in seconds
        """
        self.value = value
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def get_age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at


class LRUCache:
    """
    LRU (Least Recently Used) cache with TTL support.
    
    This cache automatically evicts least recently used items when full,
    and expires items based on TTL.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        
        logger.info(
            "LRU cache initialized",
            extra={
                "max_size": max_size,
                "default_ttl": default_ttl
            }
        )
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if entry.is_expired():
            self.expirations += 1
            self.misses += 1
            del self.cache[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override (uses default if not specified)
        """
        ttl_seconds = ttl if ttl is not None else self.default_ttl
        
        # Remove existing entry if present
        if key in self.cache:
            del self.cache[key]
        
        # Add new entry
        self.cache[key] = CacheEntry(value, ttl_seconds)
        
        # Evict oldest if over max size
        if len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.evictions += 1
    
    def delete(self, key: str):
        """Delete entry from cache."""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": hit_rate,
            "evictions": self.evictions,
            "expirations": self.expirations
        }
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
            self.expirations += 1
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)


class CacheManager:
    """
    Manages multiple caches for different data types.
    
    Provides separate caches with appropriate TTLs for:
    - Policy documents (long TTL)
    - Ticket data (short TTL)
    - Configuration (medium TTL)
    """
    
    def __init__(self):
        """Initialize cache manager with separate caches."""
        # Policy documents - long TTL (1 hour)
        self.policy_cache = LRUCache(max_size=100, default_ttl=3600.0)
        
        # Ticket data - short TTL (5 minutes)
        self.ticket_cache = LRUCache(max_size=500, default_ttl=300.0)
        
        # Configuration - medium TTL (15 minutes)
        self.config_cache = LRUCache(max_size=50, default_ttl=900.0)
        
        # Booking data - short TTL (5 minutes)
        self.booking_cache = LRUCache(max_size=500, default_ttl=300.0)
        
        logger.info("Cache manager initialized with multiple caches")
    
    # Policy cache methods
    
    def get_policy(self, policy_key: str) -> Optional[Any]:
        """Get policy document from cache."""
        return self.policy_cache.get(policy_key)
    
    def set_policy(self, policy_key: str, policy_data: Any):
        """Cache policy document."""
        self.policy_cache.set(policy_key, policy_data)
    
    # Ticket cache methods
    
    def get_ticket(self, ticket_id: str) -> Optional[Any]:
        """Get ticket data from cache."""
        return self.ticket_cache.get(f"ticket:{ticket_id}")
    
    def set_ticket(self, ticket_id: str, ticket_data: Any):
        """Cache ticket data."""
        self.ticket_cache.set(f"ticket:{ticket_id}", ticket_data)
    
    def invalidate_ticket(self, ticket_id: str):
        """Invalidate cached ticket data."""
        self.ticket_cache.delete(f"ticket:{ticket_id}")
    
    # Configuration cache methods
    
    def get_config(self, config_key: str) -> Optional[Any]:
        """Get configuration from cache."""
        return self.config_cache.get(config_key)
    
    def set_config(self, config_key: str, config_data: Any):
        """Cache configuration."""
        self.config_cache.set(config_key, config_data)
    
    # Booking cache methods
    
    def get_booking(self, booking_id: str) -> Optional[Any]:
        """Get booking data from cache."""
        return self.booking_cache.get(f"booking:{booking_id}")
    
    def set_booking(self, booking_id: str, booking_data: Any):
        """Cache booking data."""
        self.booking_cache.set(f"booking:{booking_id}", booking_data)
    
    # Management methods
    
    def cleanup_all_expired(self) -> Dict[str, int]:
        """
        Clean up expired entries from all caches.
        
        Returns:
            Dictionary with cleanup counts per cache
        """
        return {
            "policy": self.policy_cache.cleanup_expired(),
            "ticket": self.ticket_cache.cleanup_expired(),
            "config": self.config_cache.cleanup_expired(),
            "booking": self.booking_cache.cleanup_expired()
        }
    
    def clear_all(self):
        """Clear all caches."""
        self.policy_cache.clear()
        self.ticket_cache.clear()
        self.config_cache.clear()
        self.booking_cache.clear()
        logger.info("All caches cleared")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches.
        
        Returns:
            Dictionary with stats for each cache
        """
        return {
            "policy_cache": self.policy_cache.get_stats(),
            "ticket_cache": self.ticket_cache.get_stats(),
            "config_cache": self.config_cache.get_stats(),
            "booking_cache": self.booking_cache.get_stats()
        }


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns:
        The global CacheManager instance
    """
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager()
    
    return _cache_manager


def clear_all_caches():
    """Clear all caches."""
    global _cache_manager
    
    if _cache_manager is not None:
        _cache_manager.clear_all()
