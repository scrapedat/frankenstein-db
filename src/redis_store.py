"""
Redis Context Store - Fast user context and MQTT message caching

Redis-based storage for user AI contexts and MQTT message caching with TTL support.
Optimized for high-speed read/write operations with automatic compression.
"""

import json
import gzip
import time
import logging
from typing import Dict, List, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class RedisContextStore:
    """
    Redis-based user context and MQTT message caching
    
    Provides fast storage for user AI contexts and MQTT message replay
    with automatic compression and TTL management.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - using in-memory fallback")
            self.redis_client = None
            self._memory_store = {}
        else:
            try:
                self.redis_client = redis.Redis(
                    host=host, 
                    port=port, 
                    db=db,
                    decode_responses=False  # Handle binary data for compression
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"🔴 Redis context store connected to {host}:{port}/{db}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using in-memory fallback")
                self.redis_client = None
                self._memory_store = {}

    async def store_user_context(self, user_id: str, context_data: Dict, ttl: int = 3600):
        """
        Store user AI context with TTL
        
        Args:
            user_id: Unique user identifier
            context_data: Context data to store
            ttl: Time to live in seconds (default 1 hour)
        """
        key = f"user_context:{user_id}"
        
        try:
            # Compress the context data
            json_data = json.dumps(context_data, ensure_ascii=False)
            compressed = gzip.compress(json_data.encode('utf-8'))
            
            if self.redis_client:
                # Store in Redis with expiration
                self.redis_client.setex(key, ttl, compressed)
            else:
                # Fallback to memory store
                self._memory_store[key] = {
                    'data': compressed,
                    'expires': time.time() + ttl
                }
                
            logger.debug(f"Stored user context for {user_id} (TTL: {ttl}s)")
            
        except Exception as e:
            logger.error(f"Failed to store user context: {e}")

    async def get_user_context(self, user_id: str) -> Optional[Dict]:
        """
        Retrieve user context
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            User context data or None if not found/expired
        """
        key = f"user_context:{user_id}"
        
        try:
            compressed_data = None
            
            if self.redis_client:
                compressed_data = self.redis_client.get(key)
            else:
                # Check memory store
                if key in self._memory_store:
                    entry = self._memory_store[key]
                    if time.time() < entry['expires']:
                        compressed_data = entry['data']
                    else:
                        # Expired - remove it
                        del self._memory_store[key]
            
            if compressed_data:
                # Decompress and parse
                decompressed = gzip.decompress(compressed_data)
                return json.loads(decompressed.decode('utf-8'))
                
        except Exception as e:
            logger.warning(f"Failed to retrieve user context: {e}")
        
        return None

    async def cache_mqtt_message(self, topic: str, message: Dict, ttl: int = 300):
        """
        Cache MQTT messages for replay/debugging
        
        Args:
            topic: MQTT topic
            message: Message data
            ttl: Time to live in seconds (default 5 minutes)
        """
        key = f"mqtt_cache:{topic}:{int(time.time())}"
        
        try:
            json_data = json.dumps(message, ensure_ascii=False)
            
            if self.redis_client:
                self.redis_client.setex(key, ttl, json_data)
            else:
                self._memory_store[key] = {
                    'data': json_data.encode('utf-8'),
                    'expires': time.time() + ttl
                }
                
            logger.debug(f"Cached MQTT message for topic {topic}")
            
        except Exception as e:
            logger.error(f"Failed to cache MQTT message: {e}")

    async def get_recent_mqtt_messages(self, topic_pattern: str, count: int = 100) -> List[Dict]:
        """
        Get recent MQTT messages matching pattern
        
        Args:
            topic_pattern: Topic pattern to match
            count: Maximum number of messages to return
            
        Returns:
            List of messages ordered by timestamp (newest first)
        """
        messages = []
        
        try:
            if self.redis_client:
                # Use Redis pattern matching
                pattern = f"mqtt_cache:{topic_pattern}:*"
                keys = self.redis_client.keys(pattern)
                keys.sort(reverse=True)  # Most recent first
                
                for key in keys[:count]:
                    try:
                        data = self.redis_client.get(key)
                        if data:
                            message = json.loads(data.decode('utf-8') if isinstance(data, bytes) else data)
                            messages.append(message)
                    except Exception as e:
                        logger.warning(f"Failed to parse cached message: {e}")
            else:
                # Use memory store
                pattern = f"mqtt_cache:{topic_pattern}:"
                matching_keys = [k for k in self._memory_store.keys() if k.startswith(pattern)]
                matching_keys.sort(reverse=True)
                
                current_time = time.time()
                for key in matching_keys[:count]:
                    entry = self._memory_store[key]
                    if current_time < entry['expires']:
                        try:
                            data = entry['data']
                            message = json.loads(data.decode('utf-8') if isinstance(data, bytes) else data)
                            messages.append(message)
                        except Exception as e:
                            logger.warning(f"Failed to parse cached message: {e}")
                    else:
                        # Clean up expired entry
                        del self._memory_store[key]
                        
        except Exception as e:
            logger.error(f"Failed to retrieve MQTT messages: {e}")
        
        return messages

    async def store_session_data(self, session_id: str, data: Dict, ttl: int = 7200):
        """
        Store session data with TTL
        
        Args:
            session_id: Session identifier
            data: Session data
            ttl: Time to live in seconds (default 2 hours)
        """
        key = f"session:{session_id}"
        await self.store_user_context(session_id.replace('session:', ''), data, ttl)

    async def get_session_data(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve session data
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found/expired
        """
        return await self.get_user_context(session_id.replace('session:', ''))

    async def clear_user_context(self, user_id: str) -> bool:
        """
        Clear user context
        
        Args:
            user_id: User identifier
            
        Returns:
            True if context was cleared, False otherwise
        """
        key = f"user_context:{user_id}"
        
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                if key in self._memory_store:
                    del self._memory_store[key]
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to clear user context: {e}")
            return False

    def get_stats(self) -> Dict:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'redis_available': self.redis_client is not None,
            'backend': 'redis' if self.redis_client else 'memory'
        }
        
        try:
            if self.redis_client:
                info = self.redis_client.info()
                stats.update({
                    'total_keys': info.get('db0', {}).get('keys', 0),
                    'memory_usage': info.get('used_memory_human', 'unknown'),
                    'connected_clients': info.get('connected_clients', 0)
                })
            else:
                # Memory store stats
                current_time = time.time()
                active_keys = sum(1 for entry in self._memory_store.values() 
                                if current_time < entry['expires'])
                stats.update({
                    'total_keys': len(self._memory_store),
                    'active_keys': active_keys,
                    'expired_keys': len(self._memory_store) - active_keys
                })
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            stats['error'] = str(e)
        
        return stats

    def close(self):
        """Clean up resources"""
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass
        self._memory_store = {}