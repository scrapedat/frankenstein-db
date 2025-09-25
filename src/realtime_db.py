"""
realtime_db.py - High-performance real-time database with AI integration

Combines Redis for real-time operations with persistent storage
for website DNA and structural data.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any
import aioredis
from .website_dna import WebsiteDNA, generate_structure_hash

logger = logging.getLogger(__name__)

class RealTimeDB:
    """
    Real-time database implementation optimized for AI-driven updates
    and rapid data structuring operations.
    """
    
    def __init__(self, 
                redis_host: str = 'localhost',
                redis_port: int = 6379,
                redis_db: int = 0):
        """
        Initialize the real-time database
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number for DNA storage
        """
        self.redis = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        
    async def connect(self):
        """Establish Redis connection"""
        self.redis = await aioredis.create_redis_pool(
            f'redis://{self.redis_host}:{self.redis_port}',
            db=self.redis_db,
            encoding='utf-8'
        )
        logger.info("🚀 RealTimeDB connected to Redis")

    async def store_dna(self, dna: WebsiteDNA) -> bool:
        """
        Store website DNA with real-time indexing
        
        Args:
            dna: WebsiteDNA instance to store
            
        Returns:
            Success status
        """
        key = f"dna:{dna.domain}:{dna.timestamp}"
        compressed = dna.compress()
        
        # Store compressed DNA
        await self.redis.set(key, compressed)
        
        # Update domain index
        await self.redis.zadd(
            f"domain_timeline:{dna.domain}",
            dna.timestamp,
            key
        )
        
        # Index frameworks
        if dna.framework_flags:
            await self.redis.sadd(
                f"framework:{dna.framework_flags}",
                dna.domain
            )
        
        logger.info(f"📝 Stored DNA for {dna.domain} ({len(compressed)} bytes)")
        return True

    async def get_evolution_timeline(self, domain: str, hours: int = 24) -> List[WebsiteDNA]:
        """
        Get domain evolution timeline with real-time updates
        
        Args:
            domain: Domain to analyze
            hours: Hours back from current time
            
        Returns:
            List of WebsiteDNA instances
        """
        min_time = time.time() - (hours * 3600)
        
        # Get timeline keys
        keys = await self.redis.zrangebyscore(
            f"domain_timeline:{domain}",
            min_time,
            '+inf'
        )
        
        # Fetch all DNA records in parallel
        dna_data = await asyncio.gather(*[
            self.redis.get(key) for key in keys
        ])
        
        # Decompress DNA
        result = []
        for data in dna_data:
            if data:
                dna = WebsiteDNA.decompress(data)
                result.append(dna)
                
        return sorted(result, key=lambda x: x.timestamp)

    async def find_similar_structures(self, reference_hash: str, limit: int = 10) -> List[tuple]:
        """
        Find similar website structures in real-time
        
        Args:
            reference_hash: Reference structure hash
            limit: Maximum results
            
        Returns:
            List of (domain, timestamp) tuples
        """
        # TODO: Implement real-time similarity search using LSH
        # For now, return basic matches
        all_keys = await self.redis.keys("dna:*")
        results = []
        
        for key in all_keys[:100]:  # Limit initial search
            dna_data = await self.redis.get(key)
            if dna_data:
                dna = WebsiteDNA.decompress(dna_data)
                if self._calculate_hash_similarity(dna.structure_hash, reference_hash) > 0.7:
                    results.append((dna.domain, dna.timestamp))
                    
        return sorted(results, key=lambda x: x[1], reverse=True)[:limit]

    async def get_domain_stats(self, domain: str) -> Dict:
        """
        Get real-time domain statistics
        
        Args:
            domain: Domain to analyze
            
        Returns:
            Domain statistics
        """
        timeline_key = f"domain_timeline:{domain}"
        
        # Get timeline stats
        count = await self.redis.zcard(timeline_key)
        if count == 0:
            return None
            
        first = await self.redis.zrange(timeline_key, 0, 0, withscores=True)
        last = await self.redis.zrange(timeline_key, -1, -1, withscores=True)
        
        return {
            'snapshots': count,
            'first_seen': first[0][1] if first else None,
            'last_seen': last[0][1] if last else None,
            'tracking_days': (last[0][1] - first[0][1]) / 86400 if first and last else 0
        }

    async def get_framework_evolution(self, framework_flags: int, limit: int = 1000) -> List[Dict]:
        """
        Track framework adoption in real-time
        
        Args:
            framework_flags: Framework bit flags
            limit: Maximum results
            
        Returns:
            List of evolution data points
        """
        domains = await self.redis.smembers(f"framework:{framework_flags}")
        results = []
        
        for domain in domains:
            timeline = await self.get_evolution_timeline(domain, hours=720)  # 30 days
            for dna in timeline:
                if dna.framework_flags & framework_flags:
                    results.append({
                        'timestamp': dna.timestamp,
                        'domain': domain
                    })
                    
        return sorted(results, key=lambda x: x['timestamp'], reverse=True)[:limit]

    async def get_top_domains(self, limit: int = 5) -> List[str]:
        """Get most active domains"""
        all_timelines = await self.redis.keys("domain_timeline:*")
        domains = []
        
        for timeline in all_timelines:
            count = await self.redis.zcard(timeline)
            domain = timeline.split(":")[1]
            domains.append((domain, count))
            
        return [d[0] for d in sorted(domains, key=lambda x: x[1], reverse=True)[:limit]]

    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes"""
        if len(hash1) != len(hash2):
            return 0.0
            
        matches = sum(1 for a, b in zip(hash1, hash2) if a == b)
        return matches / len(hash1)

    async def close(self):
        """Clean up resources"""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
        logger.info("💤 RealTimeDB disconnected")