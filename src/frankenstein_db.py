"""
FrankensteinDB - The Beautiful Monster Database System

Main orchestrator that combines all storage systems into a unified API
for website analysis and intelligence gathering.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from .website_dna import (
    WebsiteDNA, 
    generate_structure_hash, 
    encode_frameworks, 
    decode_frameworks,
    compress_elements,
    calculate_accessibility_score,
    generate_proof_hash
)
from .scylla_emulator import ScyllaDBEmulator
from .redis_store import RedisContextStore
from .search_index import SearchIndex
from .blob_storage import BlobStorage

logger = logging.getLogger(__name__)


class FrankensteinDB:
    """
    The beautiful monster - combining all storage systems
    
    Unified interface for website intelligence gathering that orchestrates
    multiple specialized storage systems for optimal performance.
    """
    
    def __init__(self, 
                 evolution_db_path: str = "website_evolution.db",
                 search_db_path: str = "search_index.db", 
                 blob_storage_path: str = "./blob_storage",
                 redis_host: str = 'localhost',
                 redis_port: int = 6379):
        """
        Initialize FrankensteinDB with all storage components
        
        Args:
            evolution_db_path: Path for time-series evolution database
            search_db_path: Path for search index database
            blob_storage_path: Path for blob storage directory
            redis_host: Redis server host
            redis_port: Redis server port
        """
        self.dna_store = ScyllaDBEmulator(evolution_db_path)
        self.context_store = RedisContextStore(redis_host, redis_port)
        self.search_index = SearchIndex(search_db_path)
        self.blob_store = BlobStorage(blob_storage_path)
        
        logger.info("🧟‍♂️ FrankensteinDB monster is ALIVE!")

    async def store_website_snapshot(self, 
                                   url: str, 
                                   html_content: str, 
                                   structure_fingerprint: Dict, 
                                   user_context: Optional[str] = None,
                                   keywords: Optional[List[str]] = None) -> WebsiteDNA:
        """
        Store complete website snapshot across all systems
        
        Args:
            url: Website URL
            html_content: Raw HTML content
            structure_fingerprint: Analyzed website structure
            user_context: Optional user identifier for context storage
            keywords: Optional keywords for enhanced searchability
            
        Returns:
            WebsiteDNA instance representing the stored snapshot
        """
        # 1. Create Website DNA
        dna = WebsiteDNA(
            domain=self._extract_domain(url),
            timestamp=time.time(),
            structure_hash=generate_structure_hash(structure_fingerprint),
            page_type=structure_fingerprint.get('page_type', 'unknown'),
            framework_flags=encode_frameworks(structure_fingerprint.get('framework_signatures', [])),
            element_signature=compress_elements(structure_fingerprint.get('element_counts', {})),
            accessibility_score=calculate_accessibility_score(structure_fingerprint),
            performance_hints=structure_fingerprint.get('performance_hints', {}),
            proof_hash=generate_proof_hash(url, html_content)
        )
        
        # 2. Extract content description for search
        content_description = self._extract_content_description(html_content)
        
        # 3. Store in parallel across all systems
        tasks = [
            # Store DNA in time-series DB
            self.dna_store.store_dna(dna),
            
            # Index for search
            self.search_index.index_website(dna, content_description, keywords),
            
            # Store raw HTML in blob storage
            self.blob_store.store_website_content(url, html_content)
        ]
        
        # Store user context if provided
        if user_context:
            tasks.append(
                self.context_store.store_user_context(user_context, {
                    'last_url': url,
                    'timestamp': dna.timestamp,
                    'structure_hash': dna.structure_hash,
                    'domain': dna.domain
                })
            )
        
        await asyncio.gather(*tasks)
        
        logger.info(f"🧬 Stored website snapshot for {url} ({len(dna.compress())} bytes DNA)")
        return dna

    async def query_website_evolution(self, domain: str, hours: int = 24) -> List[Dict]:
        """
        Query how a website has evolved over time
        
        Args:
            domain: Domain to analyze
            hours: Hours back from current time
            
        Returns:
            List of evolution snapshots with decoded information
        """
        evolution = await self.dna_store.get_evolution_timeline(domain, hours)
        
        result = []
        for dna in evolution:
            result.append({
                'timestamp': dna.timestamp,
                'structure_hash': dna.structure_hash,
                'page_type': dna.page_type,
                'frameworks': decode_frameworks(dna.framework_flags),
                'accessibility_score': dna.accessibility_score,
                'compressed_size': len(dna.compress()),
                'proof_hash': dna.proof_hash
            })
        
        return result

    async def find_similar_websites(self, reference_url: str, limit: int = 10) -> List[Dict]:
        """
        Find websites with similar structure to reference
        
        Args:
            reference_url: Reference website URL
            limit: Maximum number of similar sites to return
            
        Returns:
            List of similar websites with metadata
        """
        # Get reference structure hash
        domain = self._extract_domain(reference_url)
        evolution = await self.dna_store.get_evolution_timeline(domain, hours=1)
        
        if not evolution:
            logger.warning(f"No evolution data found for {reference_url}")
            return []
        
        reference_hash = evolution[0].structure_hash
        similar = await self.dna_store.find_similar_structures(reference_hash, limit)
        
        # Enhance with additional metadata
        result = []
        for domain, timestamp in similar:
            domain_stats = await self.dna_store.get_domain_stats(domain)
            result.append({
                'domain': domain,
                'last_seen': timestamp,
                'similarity_hash': reference_hash[:8],
                'stats': domain_stats
            })
        
        return result

    async def search_websites(self, query: str, filters: Optional[Dict] = None, limit: int = 20) -> List[Dict]:
        """
        Search websites by content and structure
        
        Args:
            query: Search query string
            filters: Optional search filters
            limit: Maximum number of results
            
        Returns:
            List of search results with enhanced metadata
        """
        return await self.search_index.search(query, limit, filters)

    async def get_cached_content(self, url: str) -> Optional[str]:
        """
        Retrieve cached HTML content
        
        Args:
            url: Website URL
            
        Returns:
            Cached HTML content or None if not found
        """
        return await self.blob_store.get_website_content(url)

    async def get_user_context(self, user_id: str) -> Optional[Dict]:
        """
        Retrieve user context and session data
        
        Args:
            user_id: User identifier
            
        Returns:
            User context data or None if not found
        """
        return await self.context_store.get_user_context(user_id)

    async def store_user_session(self, user_id: str, session_data: Dict, ttl: int = 7200):
        """
        Store user session data
        
        Args:
            user_id: User identifier
            session_data: Session data to store
            ttl: Time to live in seconds
        """
        await self.context_store.store_user_context(user_id, session_data, ttl)

    async def get_domain_intelligence(self, domain: str) -> Dict:
        """
        Get comprehensive intelligence for a domain
        
        Args:
            domain: Domain to analyze
            
        Returns:
            Comprehensive domain intelligence report
        """
        # Gather data from all systems in parallel
        evolution_task = self.query_website_evolution(domain, hours=168)  # 1 week
        stats_task = self.dna_store.get_domain_stats(domain)
        search_task = self.search_websites(f"domain:{domain}", limit=5)
        
        evolution, stats, search_results = await asyncio.gather(
            evolution_task, stats_task, search_task, return_exceptions=True
        )
        
        # Handle any errors gracefully
        if isinstance(evolution, Exception):
            evolution = []
        if isinstance(stats, Exception):
            stats = None
        if isinstance(search_results, Exception):
            search_results = []
        
        # Analyze trends
        trends = self._analyze_evolution_trends(evolution) if evolution else {}
        
        return {
            'domain': domain,
            'statistics': stats,
            'evolution_timeline': evolution[-10:] if evolution else [],  # Last 10 snapshots
            'trends': trends,
            'search_results': search_results,
            'intelligence_timestamp': time.time()
        }

    async def get_framework_trends(self, framework: str, days: int = 30) -> Dict:
        """
        Analyze framework adoption trends
        
        Args:
            framework: Framework to analyze
            days: Days back to analyze
            
        Returns:
            Framework trend analysis
        """
        # Get framework flag for the framework
        framework_flags = encode_frameworks([framework])
        if not framework_flags:
            return {'error': f'Unknown framework: {framework}'}
        
        evolution_data = await self.dna_store.get_framework_evolution(framework_flags, limit=1000)
        websites_using = await self.search_index.search_by_framework(framework, limit=100)
        
        return {
            'framework': framework,
            'total_sites_using': len(websites_using),
            'recent_adoptions': len([d for d in evolution_data if d['timestamp'] > time.time() - (days * 86400)]),
            'evolution_data': evolution_data[:50],  # Recent 50 entries
            'top_sites': websites_using[:10]  # Top 10 sites using this framework
        }

    async def get_system_health(self) -> Dict:
        """
        Get overall system health and statistics
        
        Returns:
            System health report
        """
        # Gather stats from all components
        tasks = [
            self.dna_store.get_top_domains(limit=5),
            self.search_index.get_search_stats(),
            self.blob_store.get_storage_stats(),
            self.context_store.get_stats()
        ]
        
        top_domains, search_stats, storage_stats, context_stats = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'timestamp': time.time(),
            'components': {
                'evolution_db': 'healthy' if not isinstance(top_domains, Exception) else 'error',
                'search_index': 'healthy' if not isinstance(search_stats, Exception) else 'error',
                'blob_storage': 'healthy' if not isinstance(storage_stats, Exception) else 'error',
                'context_store': 'healthy' if not isinstance(context_stats, Exception) else 'error'
            },
            'statistics': {
                'top_domains': top_domains if not isinstance(top_domains, Exception) else [],
                'search_stats': search_stats if not isinstance(search_stats, Exception) else {},
                'storage_stats': storage_stats if not isinstance(storage_stats, Exception) else {},
                'context_stats': context_stats if not isinstance(context_stats, Exception) else {}
            }
        }

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        return urlparse(url).netloc

    def _extract_content_description(self, html: str) -> str:
        """Extract brief content description for search indexing"""
        import re
        
        # Extract title
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        title = title_match.group(1) if title_match else ""
        
        # Extract meta description
        desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
        description = desc_match.group(1) if desc_match else ""
        
        # Get first few words from body text (very basic)
        text_content = re.sub(r'<[^>]+>', ' ', html)
        words = text_content.split()[:30]  # First 30 words
        
        return f"{title} {description} {' '.join(words)}"[:500]  # Limit to 500 chars

    def _analyze_evolution_trends(self, evolution: List[Dict]) -> Dict:
        """Analyze evolution data for trends"""
        if len(evolution) < 2:
            return {'trend': 'insufficient_data'}
        
        # Sort by timestamp
        sorted_evolution = sorted(evolution, key=lambda x: x['timestamp'])
        
        # Analyze accessibility trend
        accessibility_scores = [e['accessibility_score'] for e in sorted_evolution]
        acc_trend = 'improving' if accessibility_scores[-1] > accessibility_scores[0] else 'declining'
        
        # Analyze framework changes
        framework_changes = []
        prev_frameworks = set()
        for snapshot in sorted_evolution:
            current_frameworks = set(snapshot['frameworks'])
            if prev_frameworks and current_frameworks != prev_frameworks:
                framework_changes.append({
                    'timestamp': snapshot['timestamp'],
                    'added': list(current_frameworks - prev_frameworks),
                    'removed': list(prev_frameworks - current_frameworks)
                })
            prev_frameworks = current_frameworks
        
        return {
            'accessibility_trend': acc_trend,
            'framework_changes': framework_changes,
            'total_snapshots': len(evolution),
            'time_span_hours': (sorted_evolution[-1]['timestamp'] - sorted_evolution[0]['timestamp']) / 3600
        }

    async def close(self):
        """Clean up all resources"""
        logger.info("🧟‍♂️ FrankensteinDB monster going to sleep...")
        
        # Close all components
        self.dna_store.close()
        self.search_index.close()
        self.context_store.close()
        
        # Blob storage doesn't need explicit closing
        
        logger.info("💤 FrankensteinDB resources cleaned up")