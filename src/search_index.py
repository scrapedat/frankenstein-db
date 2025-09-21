"""
Search Index - Full-text search using SQLite FTS5

Fast content and metadata search capabilities using SQLite's FTS5
virtual table engine for website discovery and analysis.
"""

import asyncio
import sqlite3
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from .website_dna import WebsiteDNA, decode_frameworks

logger = logging.getLogger(__name__)


class SearchIndex:
    """
    Simple search index using SQLite FTS5
    
    Provides full-text search capabilities for website content,
    frameworks, and metadata with relevance ranking.
    """
    
    def __init__(self, db_path: str = "search_index.db"):
        self.db_path = db_path
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._init_schema()

    def _init_schema(self):
        """Initialize FTS5 search schema with optimized configuration"""
        with sqlite3.connect(self.db_path) as conn:
            # Enable FTS5 if not available
            try:
                conn.execute("SELECT fts5(?)", ("test",))
            except sqlite3.OperationalError:
                logger.error("SQLite FTS5 not available - search functionality will be limited")
                return
            
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS website_search USING fts5(
                    domain,
                    page_type,
                    frameworks,
                    content_description,
                    accessibility_features,
                    keywords,
                    timestamp UNINDEXED,
                    accessibility_score UNINDEXED,
                    framework_flags UNINDEXED
                )
            ''')
            
            # Create auxiliary table for additional metadata
            conn.execute('''
                CREATE TABLE IF NOT EXISTS search_metadata (
                    domain TEXT PRIMARY KEY,
                    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_pages INTEGER DEFAULT 1,
                    avg_accessibility REAL,
                    dominant_framework TEXT
                )
            ''')
            
            logger.info("🔍 Search index initialized with FTS5")

    async def index_website(self, dna: WebsiteDNA, content_description: str = "", keywords: List[str] = None):
        """
        Add website to search index
        
        Args:
            dna: WebsiteDNA instance to index
            content_description: Extracted content description
            keywords: Additional keywords for searchability
        """
        def _index():
            # Convert framework flags to readable list
            frameworks = decode_frameworks(dna.framework_flags)
            
            # Prepare searchable text
            accessibility_text = f"accessibility_score_{int(dna.accessibility_score)}"
            if dna.accessibility_score >= 8.0:
                accessibility_text += " high_accessibility excellent_accessibility"
            elif dna.accessibility_score >= 6.0:
                accessibility_text += " good_accessibility"
            elif dna.accessibility_score >= 4.0:
                accessibility_text += " moderate_accessibility"
            else:
                accessibility_text += " low_accessibility poor_accessibility"
            
            keywords_text = ' '.join(keywords) if keywords else ""
            
            with sqlite3.connect(self.db_path) as conn:
                # Insert/update search entry
                conn.execute('''
                    INSERT OR REPLACE INTO website_search
                    (domain, page_type, frameworks, content_description, 
                     accessibility_features, keywords, timestamp, accessibility_score, framework_flags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    dna.domain,
                    dna.page_type,
                    ' '.join(frameworks),
                    content_description,
                    accessibility_text,
                    keywords_text,
                    int(dna.timestamp),
                    dna.accessibility_score,
                    dna.framework_flags
                ))
                
                # Update metadata
                conn.execute('''
                    INSERT OR REPLACE INTO search_metadata
                    (domain, last_indexed, total_pages, avg_accessibility, dominant_framework)
                    VALUES (?, CURRENT_TIMESTAMP, 
                           COALESCE((SELECT total_pages FROM search_metadata WHERE domain = ?) + 1, 1),
                           ?, ?)
                ''', (
                    dna.domain,
                    dna.domain,
                    dna.accessibility_score,
                    frameworks[0] if frameworks else 'none'
                ))
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _index)

    async def search(self, query: str, limit: int = 20, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search websites by content and metadata
        
        Args:
            query: Search query string
            limit: Maximum number of results
            filters: Optional filters (framework, accessibility_min, page_type)
            
        Returns:
            List of search results with relevance scores
        """
        def _search():
            with sqlite3.connect(self.db_path) as conn:
                # Build search query with filters
                base_query = '''
                    SELECT domain, page_type, frameworks, timestamp, 
                           accessibility_score, rank, content_description
                    FROM website_search 
                    WHERE website_search MATCH ?
                '''
                
                params = [query]
                
                # Add filters
                if filters:
                    conditions = []
                    
                    if 'framework' in filters:
                        conditions.append("frameworks LIKE ?")
                        params.append(f"%{filters['framework']}%")
                    
                    if 'accessibility_min' in filters:
                        conditions.append("accessibility_score >= ?")
                        params.append(filters['accessibility_min'])
                    
                    if 'page_type' in filters:
                        conditions.append("page_type = ?")
                        params.append(filters['page_type'])
                    
                    if conditions:
                        base_query += " AND " + " AND ".join(conditions)
                
                base_query += " ORDER BY rank LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(base_query, params)
                
                return [
                    {
                        'domain': row[0],
                        'page_type': row[1], 
                        'frameworks': row[2].split() if row[2] else [],
                        'timestamp': row[3],
                        'accessibility_score': row[4],
                        'relevance': float(row[5]) if row[5] else 0.0,
                        'content_preview': row[6][:200] + '...' if len(row[6]) > 200 else row[6]
                    }
                    for row in cursor.fetchall()
                ]
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _search)

    async def search_by_framework(self, framework: str, limit: int = 50) -> List[Dict]:
        """
        Search websites using a specific framework
        
        Args:
            framework: Framework name (e.g., 'react', 'vue')
            limit: Maximum number of results
            
        Returns:
            List of websites using the framework
        """
        def _search():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT domain, page_type, frameworks, timestamp, accessibility_score
                    FROM website_search 
                    WHERE frameworks LIKE ?
                    ORDER BY timestamp DESC LIMIT ?
                ''', (f"%{framework}%", limit))
                
                return [
                    {
                        'domain': row[0],
                        'page_type': row[1], 
                        'frameworks': row[2].split() if row[2] else [],
                        'timestamp': row[3],
                        'accessibility_score': row[4]
                    }
                    for row in cursor.fetchall()
                ]
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _search)

    async def get_popular_frameworks(self, limit: int = 10) -> List[Dict]:
        """
        Get most popular frameworks by usage count
        
        Args:
            limit: Maximum number of frameworks to return
            
        Returns:
            List of frameworks with usage statistics
        """
        def _query():
            with sqlite3.connect(self.db_path) as conn:
                # This is a simplified approach - in practice you'd want more sophisticated counting
                cursor = conn.execute('''
                    SELECT dominant_framework, COUNT(*) as usage_count,
                           AVG(avg_accessibility) as avg_accessibility
                    FROM search_metadata 
                    WHERE dominant_framework != 'none'
                    GROUP BY dominant_framework
                    ORDER BY usage_count DESC LIMIT ?
                ''', (limit,))
                
                return [
                    {
                        'framework': row[0],
                        'usage_count': row[1],
                        'avg_accessibility': round(row[2], 2) if row[2] else 0
                    }
                    for row in cursor.fetchall()
                ]
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _query)

    async def suggest_domains(self, partial_domain: str, limit: int = 10) -> List[str]:
        """
        Get domain suggestions based on partial input
        
        Args:
            partial_domain: Partial domain string
            limit: Maximum suggestions
            
        Returns:
            List of domain suggestions
        """
        def _query():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT DISTINCT domain FROM website_search 
                    WHERE domain LIKE ?
                    ORDER BY domain LIMIT ?
                ''', (f"{partial_domain}%", limit))
                
                return [row[0] for row in cursor.fetchall()]
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _query)

    async def get_search_stats(self) -> Dict:
        """
        Get search index statistics
        
        Returns:
            Dictionary with index statistics
        """
        def _query():
            with sqlite3.connect(self.db_path) as conn:
                # Get basic stats
                cursor = conn.execute('SELECT COUNT(*) FROM website_search')
                total_entries = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT COUNT(DISTINCT domain) FROM website_search')
                unique_domains = cursor.fetchone()[0]
                
                cursor = conn.execute('''
                    SELECT AVG(accessibility_score), MAX(accessibility_score), MIN(accessibility_score)
                    FROM website_search
                ''')
                acc_stats = cursor.fetchone()
                
                return {
                    'total_entries': total_entries,
                    'unique_domains': unique_domains,
                    'avg_accessibility': round(acc_stats[0], 2) if acc_stats[0] else 0,
                    'max_accessibility': acc_stats[1] if acc_stats[1] else 0,
                    'min_accessibility': acc_stats[2] if acc_stats[2] else 0
                }
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _query)

    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)