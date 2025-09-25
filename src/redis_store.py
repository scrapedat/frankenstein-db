"""
Redis Context Store - Fast user context and MQTT message caching

Redis-based storage for user AI contexts and MQTT message caching with TTL support.
Optimized for high-speed read/write operations with automatic compression.
"""

import json
import gzip
import time
import logging
from typing import Dict, List, Optional, Any

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = type('MockRedis', (), {})()  # Create a mock object

logger = logging.getLogger(__name__)


class RedisContextStore:
    """
    Enhanced Redis-based multi-purpose storage system

    Provides fast storage for:
    - User AI contexts and sessions (DB 0)
    - AI knowledge base (DB 1)
    - Scraping logic and patterns (DB 2)
    - Website DNA cache (DB 3)
    - Session data and temporary storage (DB 4)

    Supports automatic compression, TTL management, and multi-database architecture.
    """

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.default_db = db

        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - using in-memory fallback")
            self.redis_clients = {}
            self._memory_store = {}
        else:
            self.redis_clients = {}
            # Initialize clients for different databases
            self._init_redis_clients()

        # Performance monitoring
        self._perf_metrics = {
            'operations': {},
            'latency': {},
            'errors': {},
            'cache_hits': 0,
            'cache_misses': 0
        }
        self._start_time = time.time()

    def _init_redis_clients(self):
        """Initialize Redis clients for different databases"""
        if not REDIS_AVAILABLE:
            return

        databases = {
            0: 'user_contexts',
            1: 'ai_knowledge',
            2: 'scraping_logic',
            3: 'website_dna_cache',
            4: 'session_data'
        }

        for db_num, db_name in databases.items():
            try:
                client = redis.Redis(  # type: ignore
                    host=self.host,
                    port=self.port,
                    db=db_num,
                    decode_responses=False  # Handle binary data for compression
                )
                # Test connection
                client.ping()
                self.redis_clients[db_num] = client
                logger.info(f"🔴 Redis DB {db_num} ({db_name}) connected")
            except (AttributeError, Exception) as e:
                logger.warning(f"Redis DB {db_num} ({db_name}) failed: {e}")
                self.redis_clients[db_num] = None

        # Set default client
        self.redis_client = self.redis_clients.get(self.default_db)

        # Fallback to in-memory if no Redis available
        if not any(self.redis_clients.values()):
            logger.warning("No Redis databases available - using in-memory fallback")
            self._memory_store = {}

        # Performance monitoring
        self._perf_metrics = {
            'operations': {},
            'latency': {},
            'errors': {},
            'cache_hits': 0,
            'cache_misses': 0
        }
        self._start_time = time.time()

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

    def _get_client(self, db_num: int):
        """Get Redis client for specific database"""
        if not REDIS_AVAILABLE:
            return None
        return self.redis_clients.get(db_num)

    async def store_ai_knowledge(self, knowledge_id: str, knowledge_data: Dict, ttl: int = 86400):
        """
        Store AI knowledge patterns and learned data

        Args:
            knowledge_id: Unique knowledge identifier
            knowledge_data: Knowledge data (patterns, rules, etc.)
            ttl: Time to live in seconds (default 24 hours)
        """
        await self._store_in_db(1, f"ai:{knowledge_id}", knowledge_data, ttl)

    async def get_ai_knowledge(self, knowledge_id: str) -> Optional[Dict]:
        """
        Retrieve AI knowledge

        Args:
            knowledge_id: Knowledge identifier

        Returns:
            Knowledge data or None if not found
        """
        return await self._get_from_db(1, f"ai:{knowledge_id}")

    async def store_scraping_logic(self, site_pattern: str, scraping_config: Dict, ttl: int = 604800):
        """
        Store scraping logic and selectors for websites

        Args:
            site_pattern: Site pattern or domain
            scraping_config: Scraping configuration (selectors, patterns, etc.)
            ttl: Time to live in seconds (default 1 week)
        """
        await self._time_operation('store_scraping_logic',
                                 self._store_in_db(2, f"scrape:{site_pattern}", scraping_config, ttl))

    async def get_scraping_logic(self, site_pattern: str) -> Optional[Dict]:
        """
        Retrieve scraping logic for a site

        Args:
            site_pattern: Site pattern or domain

        Returns:
            Scraping configuration or None if not found
        """
        return await self._get_from_db(2, f"scrape:{site_pattern}")

    async def cache_website_dna(self, url_hash: str, dna_data: Dict, ttl: int = 3600):
        """
        Cache website DNA analysis results

        Args:
            url_hash: Hash of the URL
            dna_data: DNA analysis data
            ttl: Time to live in seconds (default 1 hour)
        """
        await self._store_in_db(3, f"dna:{url_hash}", dna_data, ttl)

    async def get_cached_dna(self, url_hash: str) -> Optional[Dict]:
        """
        Retrieve cached website DNA

        Args:
            url_hash: Hash of the URL

        Returns:
            DNA data or None if not found
        """
        return await self._get_from_db(3, f"dna:{url_hash}")

    async def store_session_metadata(self, session_id: str, metadata: Dict, ttl: int = 7200):
        """
        Store session metadata and analysis results

        Args:
            session_id: Session identifier
            metadata: Session metadata
            ttl: Time to live in seconds (default 2 hours)
        """
        await self._store_in_db(4, f"session:{session_id}", metadata, ttl)

    async def get_session_metadata(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve session metadata

        Args:
            session_id: Session identifier

        Returns:
            Session metadata or None if not found
        """
        return await self._get_from_db(4, f"session:{session_id}")

    async def _store_in_db(self, db_num: int, key: str, data: Dict, ttl: int):
        """Store data in specific Redis database"""
        client = self._get_client(db_num)
        if client:
            try:
                json_data = json.dumps(data, ensure_ascii=False)
                compressed = gzip.compress(json_data.encode('utf-8'))
                client.setex(key, ttl, compressed)
                logger.debug(f"Stored in DB {db_num}: {key}")
            except Exception as e:
                logger.error(f"Failed to store in DB {db_num}: {e}")
        else:
            # Fallback to memory
            mem_key = f"{db_num}:{key}"
            self._memory_store[mem_key] = {
                'data': gzip.compress(json.dumps(data, ensure_ascii=False).encode('utf-8')),
                'expires': time.time() + ttl
            }

    async def _get_from_db(self, db_num: int, key: str) -> Optional[Dict]:
        """Retrieve data from specific Redis database"""
        client = self._get_client(db_num)
        if client:
            try:
                compressed_data = client.get(key)
                if compressed_data:
                    decompressed = gzip.decompress(compressed_data)
                    return json.loads(decompressed.decode('utf-8'))
            except Exception as e:
                logger.warning(f"Failed to retrieve from DB {db_num}: {e}")
        else:
            # Check memory fallback
            mem_key = f"{db_num}:{key}"
            if mem_key in self._memory_store:
                entry = self._memory_store[mem_key]
                if time.time() < entry['expires']:
                    try:
                        decompressed = gzip.decompress(entry['data'])
                        return json.loads(decompressed.decode('utf-8'))
                    except Exception as e:
                        logger.warning(f"Failed to retrieve from memory: {e}")
                else:
                    del self._memory_store[mem_key]

        return None

    async def atomic_config_sync(self, website: str, config_data: Dict[str, Any], ttls: Dict[str, int]) -> bool:
        """
        Atomically sync scraping configuration across multiple Redis databases

        Args:
            website: Website domain
            config_data: Configuration data with keys: scraping_logic, dna, ai_metadata, version, last_updated
            ttls: TTL values for each data type

        Returns:
            True if all operations succeeded atomically, False otherwise
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available for atomic sync")
            return False

        try:
            # Get all required clients
            scraping_client = self._get_client(2)  # scraping_logic
            dna_client = self._get_client(3)        # website_dna_cache
            ai_client = self._get_client(1)         # ai_knowledge

            if not all([scraping_client, dna_client, ai_client]):
                logger.warning("Not all Redis clients available for atomic sync")
                return False

            # Prepare data for compression
            scraping_data = json.dumps(config_data['scraping_logic'], ensure_ascii=False)
            dna_data = json.dumps(config_data['dna'], ensure_ascii=False)
            ai_data = json.dumps(config_data['ai_metadata'], ensure_ascii=False)

            # Compress data
            compressed_scraping = gzip.compress(scraping_data.encode('utf-8'))
            compressed_dna = gzip.compress(dna_data.encode('utf-8'))
            compressed_ai = gzip.compress(ai_data.encode('utf-8'))

            # Use pipelines for atomic operations
            if scraping_client and dna_client and ai_client:
                scraping_pipeline = scraping_client.pipeline()  # type: ignore
                dna_pipeline = dna_client.pipeline()  # type: ignore
                ai_pipeline = ai_client.pipeline()  # type: ignore
            else:
                return False

            # Queue operations in pipelines
            scraping_key = f"scrape:{website}"
            dna_key = f"dna:{website}"
            ai_key = f"ai:config:{website}"

            scraping_pipeline.setex(scraping_key, ttls['scraping_logic'], compressed_scraping)
            dna_pipeline.setex(dna_key, ttls['dna'], compressed_dna)
            ai_pipeline.setex(ai_key, ttls['ai_metadata'], compressed_ai)

            # Execute all pipelines atomically
            scraping_result = scraping_pipeline.execute()
            dna_result = dna_pipeline.execute()
            ai_result = ai_pipeline.execute()

            # Check if all operations succeeded
            success = all([
                all(scraping_result),  # All scraping operations succeeded
                all(dna_result),       # All DNA operations succeeded
                all(ai_result)         # All AI operations succeeded
            ])

            if success:
                logger.debug(f"Atomic config sync successful for {website} v{config_data.get('version', 'unknown')}")
            else:
                logger.error(f"Atomic config sync failed for {website}: scraping={scraping_result}, dna={dna_result}, ai={ai_result}")

            return success

        except Exception as e:
            logger.error(f"Atomic config sync failed for {website}: {e}")
            return False



    def _record_operation(self, operation: str, duration: float, success: bool = True):
        """Record performance metrics for an operation"""
        if operation not in self._perf_metrics['operations']:
            self._perf_metrics['operations'][operation] = {
                'count': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'success_count': 0,
                'error_count': 0
            }

        metrics = self._perf_metrics['operations'][operation]
        metrics['count'] += 1
        metrics['total_time'] += duration
        metrics['avg_time'] = metrics['total_time'] / metrics['count']

        if success:
            metrics['success_count'] += 1
        else:
            metrics['error_count'] += 1

    async def _time_operation(self, operation_name: str, coro):
        """Time an async operation and record metrics"""
        start_time = time.time()
        try:
            result = await coro
            duration = time.time() - start_time
            self._record_operation(operation_name, duration, success=True)
            return result
        except Exception as e:
            duration = time.time() - start_time
            self._record_operation(operation_name, duration, success=False)
            raise e

    def get_performance_metrics(self) -> Dict:
        """
        Get comprehensive performance metrics

        Returns:
            Dictionary with performance statistics
        """
        uptime = time.time() - self._start_time

        metrics = {
            'uptime_seconds': uptime,
            'operations': self._perf_metrics['operations'],
            'cache_stats': {
                'hits': self._perf_metrics['cache_hits'],
                'misses': self._perf_metrics['cache_misses'],
                'hit_rate': (self._perf_metrics['cache_hits'] /
                           max(1, self._perf_metrics['cache_hits'] + self._perf_metrics['cache_misses']))
            },
            'redis_stats': self.get_enhanced_stats() if REDIS_AVAILABLE else {'redis_available': False}
        }

        return metrics

    def get_enhanced_stats(self) -> Dict:
        """
        Get enhanced statistics for all Redis databases

        Returns:
            Dictionary with statistics for each database
        """
        stats = {
            'redis_available': REDIS_AVAILABLE,
            'databases': {}
        }

        for db_num in range(5):  # Check DBs 0-4
            db_name = {
                0: 'user_contexts',
                1: 'ai_knowledge',
                2: 'scraping_logic',
                3: 'website_dna_cache',
                4: 'session_data'
            }.get(db_num, f'db_{db_num}')

            client = self._get_client(db_num)
            if client:
                try:
                    info = client.info()
                    stats['databases'][db_name] = {
                        'status': 'connected',
                        'keys': info.get('db0', {}).get('keys', 0),  # Note: info shows db0 regardless
                        'memory_usage': info.get('used_memory_human', 'unknown'),
                        'connections': info.get('connected_clients', 0),
                        'ops_per_sec': info.get('instantaneous_ops_per_sec', 0)
                    }
                except Exception as e:
                    stats['databases'][db_name] = {'status': 'error', 'error': str(e)}
            else:
                stats['databases'][db_name] = {'status': 'disconnected'}

        return stats

    def close(self):
        """Clean up resources"""
        for client in self.redis_clients.values():
            if client:
                try:
                    client.close()
                except:
                    pass
        self.redis_clients = {}
        self._memory_store = {}