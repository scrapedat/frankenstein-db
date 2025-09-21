"""
ScyllaDB Emulator - Time-series website evolution storage

SQLite-based emulator for ScyllaDB functionality, focusing on time-series
storage of website evolution data with efficient querying capabilities.
"""

import asyncio
import sqlite3
import time
import logging
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from .website_dna import WebsiteDNA

logger = logging.getLogger(__name__)


class ScyllaDBEmulator:
    """
    ScyllaDB emulator using SQLite for development
    
    Provides time-series storage optimized for website evolution tracking.
    In production, this would be replaced with actual ScyllaDB driver.
    """
    
    def __init__(self, db_path: str = "website_evolution.db"):
        self.db_path = db_path
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema with optimized indices"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS website_evolution (
                    domain TEXT,
                    timestamp INTEGER,
                    dna_hash TEXT,
                    page_type TEXT,
                    framework_flags INTEGER,
                    compressed_dna BLOB,
                    proof_hash TEXT,
                    accessibility_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (domain, timestamp)
                )
            ''')
            
            # Create indices for fast queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_domain_time ON website_evolution(domain, timestamp DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_framework ON website_evolution(framework_flags)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_page_type ON website_evolution(page_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_proof ON website_evolution(proof_hash)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_accessibility ON website_evolution(accessibility_score)')
            
            logger.info("🧬 ScyllaDB emulator schema initialized")

    async def store_dna(self, dna: WebsiteDNA):
        """
        Store website DNA with time-series partitioning
        
        Args:
            dna: WebsiteDNA instance to store
        """
        def _store():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO website_evolution 
                    (domain, timestamp, dna_hash, page_type, framework_flags, 
                     compressed_dna, proof_hash, accessibility_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    dna.domain,
                    int(dna.timestamp),
                    dna.structure_hash,
                    dna.page_type,
                    dna.framework_flags,
                    dna.compress(),
                    dna.proof_hash,
                    dna.accessibility_score
                ))
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _store)

    async def get_evolution_timeline(self, domain: str, hours: int = 24) -> List[WebsiteDNA]:
        """
        Get website evolution over time
        
        Args:
            domain: Domain name to query
            hours: Hours back from current time
            
        Returns:
            List of WebsiteDNA instances ordered by timestamp (newest first)
        """
        def _query():
            since = int(time.time()) - (hours * 3600)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT compressed_dna FROM website_evolution 
                    WHERE domain = ? AND timestamp >= ?
                    ORDER BY timestamp DESC LIMIT 100
                ''', (domain, since))
                
                results = []
                for row in cursor.fetchall():
                    try:
                        dna = WebsiteDNA.decompress(row[0])
                        results.append(dna)
                    except Exception as e:
                        logger.warning(f"Failed to decompress DNA: {e}")
                
                return results
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _query)

    async def find_similar_structures(self, structure_hash: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Find websites with similar structural hashes
        
        Args:
            structure_hash: Reference structure hash
            limit: Maximum number of results
            
        Returns:
            List of (domain, timestamp) tuples for similar structures
        """
        def _query():
            # Simple similarity: same first 8 characters of hash
            pattern = structure_hash[:8] + '%'
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT DISTINCT domain, MAX(timestamp) as latest
                    FROM website_evolution 
                    WHERE dna_hash LIKE ?
                    GROUP BY domain
                    ORDER BY latest DESC LIMIT ?
                ''', (pattern, limit))
                
                return [(row[0], row[1]) for row in cursor.fetchall()]
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _query)

    async def get_framework_evolution(self, framework_flag: int, limit: int = 1000) -> List[Dict]:
        """
        Track framework adoption over time
        
        Args:
            framework_flag: Bit flag for specific framework
            limit: Maximum number of results
            
        Returns:
            List of dictionaries with domain, timestamp, and framework flags
        """
        def _query():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT domain, timestamp, framework_flags
                    FROM website_evolution 
                    WHERE (framework_flags & ?) > 0
                    ORDER BY timestamp DESC LIMIT ?
                ''', (framework_flag, limit))
                
                return [{'domain': row[0], 'timestamp': row[1], 'flags': row[2]} 
                       for row in cursor.fetchall()]
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _query)

    async def get_domain_stats(self, domain: str) -> Optional[Dict]:
        """
        Get comprehensive statistics for a domain
        
        Args:
            domain: Domain to analyze
            
        Returns:
            Dictionary with domain statistics or None if not found
        """
        def _query():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_snapshots,
                        MIN(timestamp) as first_seen,
                        MAX(timestamp) as last_seen,
                        AVG(accessibility_score) as avg_accessibility,
                        MAX(accessibility_score) as max_accessibility,
                        COUNT(DISTINCT page_type) as unique_page_types
                    FROM website_evolution 
                    WHERE domain = ?
                ''', (domain,))
                
                row = cursor.fetchone()
                if row and row[0] > 0:
                    return {
                        'domain': domain,
                        'total_snapshots': row[0],
                        'first_seen': row[1],
                        'last_seen': row[2],
                        'avg_accessibility': round(row[3], 2) if row[3] else 0,
                        'max_accessibility': row[4] if row[4] else 0,
                        'unique_page_types': row[5]
                    }
                return None
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _query)

    async def get_top_domains(self, limit: int = 20) -> List[Dict]:
        """
        Get most actively tracked domains
        
        Args:
            limit: Maximum number of domains to return
            
        Returns:
            List of domain statistics ordered by activity
        """
        def _query():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT 
                        domain,
                        COUNT(*) as snapshot_count,
                        MAX(timestamp) as last_seen,
                        AVG(accessibility_score) as avg_accessibility
                    FROM website_evolution 
                    GROUP BY domain
                    ORDER BY snapshot_count DESC, last_seen DESC
                    LIMIT ?
                ''', (limit,))
                
                return [
                    {
                        'domain': row[0],
                        'snapshot_count': row[1],
                        'last_seen': row[2],
                        'avg_accessibility': round(row[3], 2) if row[3] else 0
                    }
                    for row in cursor.fetchall()
                ]
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _query)

    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)