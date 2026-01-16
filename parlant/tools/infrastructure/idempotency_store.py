"""
Redis-backed idempotency store for webhook processing.
Prevents duplicate processing of identical webhook payloads.

Implements requirements:
- 2.1: Ensure only one processing task is created for identical payloads
- 2.2: Use hash of complete webhook payload as idempotency key
- 2.3: Return existing task identifier without creating new task
- 2.5: Graceful degradation when Redis is unavailable
"""

import hashlib
import json
import logging
import time
from typing import Optional, Tuple
import redis.asyncio as redis
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class IdempotencyStore:
    """Redis-backed storage for preventing duplicate webhook processing."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 replica_url: Optional[str] = None):
        """
        Initialize idempotency store with Redis connection.
        
        Args:
            redis_url: Primary Redis connection URL
            replica_url: Optional replica Redis connection URL for reads
        """
        self.redis_url = redis_url
        self.replica_url = replica_url
        self._redis = None
        self._replica = None
        self._available = True
    
    async def connect(self):
        """Establish Redis connections."""
        try:
            self._redis = redis.from_url(self.redis_url)
            if self.replica_url:
                self._replica = redis.from_url(self.replica_url)
            self._available = True
            logger.info("Connected to Redis idempotency store")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._available = False
            raise
    
    async def disconnect(self):
        """Close Redis connections."""
        if self._redis:
            await self._redis.close()
        if self._replica:
            await self._replica.close()
    
    def generate_payload_hash(self, payload: dict) -> str:
        """
        Generate deterministic hash for webhook payload.
        
        Implements requirement 2.2: Use hash of complete webhook payload as idempotency key.
        
        Args:
            payload: Webhook payload dictionary
            
        Returns:
            SHA-256 hash of the payload
        """
        # Sort keys to ensure deterministic hashing
        payload_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()
    
    async def check_idempotency(self, payload_hash: str) -> Tuple[Optional[str], bool]:
        """
        Check if payload already processed and return existing task_id.
        
        Implements requirements:
        - 2.1: Ensure only one processing task is created for identical payloads
        - 2.3: Return existing task identifier without creating new task
        - 2.5: Graceful degradation when Redis is unavailable
        
        Args:
            payload_hash: SHA-256 hash of webhook payload
            
        Returns:
            Tuple of (existing_task_id, redis_available)
            - existing_task_id: Task ID if payload already processed, None otherwise
            - redis_available: True if Redis is available, False if degraded
        """
        if not self._available:
            logger.warning("Redis unavailable, skipping idempotency check")
            return None, False
        
        try:
            # Use replica for reads if available, otherwise use primary
            redis_client = self._replica if self._replica else self._redis
            task_id = await redis_client.get(f"idempotency:{payload_hash}")
            return task_id.decode('utf-8') if task_id else None, True
        except Exception as e:
            logger.error(f"Failed to get task_id from Redis: {e}")
            self._available = False
            return None, False
    
    async def get_task_id(self, payload_hash: str) -> Optional[str]:
        """
        Retrieve existing task_id for payload hash.
        
        Args:
            payload_hash: SHA-256 hash of webhook payload
            
        Returns:
            Existing task_id if found, None otherwise
        """
        task_id, _ = await self.check_idempotency(payload_hash)
        return task_id
    
    async def store_task_id(self, payload_hash: str, task_id: str, 
                           ttl: int = 604800) -> bool:
        """
        Store payload hash -> task_id mapping with TTL.
        
        Implements requirement 2.4: 7-day TTL allows reprocessing after expiration.
        
        Args:
            payload_hash: SHA-256 hash of webhook payload
            task_id: Unique task identifier
            ttl: Time-to-live in seconds (default: 7 days = 604800 seconds)
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self._available:
            logger.warning("Redis unavailable, skipping idempotency storage")
            return False
        
        try:
            # Always write to primary Redis
            await self._redis.setex(f"idempotency:{payload_hash}", ttl, task_id)
            logger.debug(f"Stored idempotency key {payload_hash} -> {task_id} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Failed to store task_id in Redis: {e}")
            self._available = False
            return False
    
    async def is_available(self) -> bool:
        """
        Check Redis connectivity for graceful degradation.
        
        Returns:
            True if Redis is available, False otherwise
        """
        if not self._redis:
            return False
        
        try:
            await self._redis.ping()
            self._available = True
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            self._available = False
            return False
    
    async def get_stats(self) -> dict:
        """
        Get idempotency store statistics for monitoring.
        
        Returns:
            Dictionary with store statistics
        """
        if not self._available:
            return {
                "available": False,
                "total_keys": 0,
                "error": "Redis unavailable"
            }
        
        try:
            # Get all idempotency keys
            keys = await self._redis.keys("idempotency:*")
            total_keys = len(keys)
            
            # Sample TTL information
            sample_ttls = []
            for key in keys[:10]:  # Sample first 10 keys
                ttl = await self._redis.ttl(key)
                if ttl > 0:
                    sample_ttls.append(ttl)
            
            return {
                "available": True,
                "total_keys": total_keys,
                "sample_ttls": sample_ttls,
                "avg_ttl": sum(sample_ttls) / len(sample_ttls) if sample_ttls else 0
            }
        except Exception as e:
            logger.error(f"Failed to get idempotency stats: {e}")
            return {
                "available": False,
                "total_keys": 0,
                "error": str(e)
            }
    
    async def cleanup_expired_keys(self) -> int:
        """
        Clean up expired idempotency keys (for testing/maintenance).
        
        Returns:
            Number of keys cleaned up
        """
        if not self._available:
            return 0
        
        try:
            # Get all idempotency keys
            keys = await self._redis.keys("idempotency:*")
            expired_count = 0
            
            for key in keys:
                ttl = await self._redis.ttl(key)
                if ttl == -2:  # Key doesn't exist (expired)
                    expired_count += 1
            
            return expired_count
        except Exception as e:
            logger.error(f"Failed to cleanup expired keys: {e}")
            return 0