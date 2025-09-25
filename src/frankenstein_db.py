"""
FrankensteinDB - The Multi-Database System

Main orchestrator that combines all storage systems into a unified API/
mqtt pub/sub for website analysis and intelligence gathering.
"""

import asyncio
import time
import logging
import sqlite3
import json
import yaml
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse
from pathlib import Path
from dataclasses import dataclass

from .website_dna import (
    WebsiteDNA, 
    generate_structure_hash, 
    encode_frameworks, 
    decode_frameworks,
    compress_elements,
    calculate_accessibility_score,
    generate_proof_hash,
    ScrapingStrategy
)
from .redis_store import RedisContextStore
from .blob_storage import BlobStorage
from .secure_mqtt import SecureMQTTClient, ComponentType, PubSubMode
from .mqtt_topics import PubSubPatterns
from .mqtt_handlers import MQTTHandlers
from .ai_workflow import AIWorkflowManager

logger = logging.getLogger(__name__)


@dataclass
class ScrapingConfig:
    """Represents a complete scraping configuration for a website"""
    website: str
    version: str
    last_updated: str
    status: str
    dna: Dict[str, Any]
    scraping_logic: Dict[str, Any]
    ai_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'website': self.website,
            'version': self.version,
            'last_updated': self.last_updated,
            'status': self.status,
            'dna': self.dna,
            'scraping_logic': self.scraping_logic,
            'ai_metadata': self.ai_metadata
        }


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
                 redis_port: int = 6379,
                 mqtt_host: str = 'localhost',
                 mqtt_port: int = 1883,
                 enable_ai_features: bool = True):
        """
        Initialize FrankensteinDB with all storage components
        
        Args:
            evolution_db_path: Path for time-series evolution database
            search_db_path: Path for search index database
            blob_storage_path: Path for blob storage directory
            redis_host: Redis server host
            redis_port: Redis server port
            mqtt_host: MQTT broker host
            mqtt_port: MQTT broker port
            enable_ai_features: Whether to enable AI features
        """
        # Core storage components
        self.context_store = RedisContextStore(redis_host, redis_port)
        self.blob_store = BlobStorage(blob_storage_path)
        
        # Initialize executors for database operations
        self.db_executor = ThreadPoolExecutor(max_workers=4)
        self.search_executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize database paths
        self.evolution_db_path = Path(evolution_db_path)
        self.search_db_path = Path(search_db_path)
        
        # Create database directories
        self.evolution_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.search_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases
        self._init_evolution_db()
        self._init_search_db()
        
        # Feature flags
        self.enable_ai_features = enable_ai_features
        
        # Initialize secure MQTT system
        self.mqtt = SecureMQTTClient(
            component_type=ComponentType.DATABASE,
            component_id='frankenstein_db',
            broker_host=mqtt_host,
            broker_port=mqtt_port,
            cert_path='/app/certs',
            default_mode=PubSubMode.WORKFLOW
        )
        
        # Configure topic modes
        topics = PubSubPatterns.get_component_topics('db', 'frankenstein_db')
        for name, config in topics.items():
            self.mqtt.set_topic_mode(
                config['pattern'],
                PubSubMode.WORKFLOW if 'ai' in name else PubSubMode.CONSTANT
            )
        
        # Initialize handlers
        self.mqtt_handlers = MQTTHandlers(self)
        
        # Initialize AI workflow manager
        if self.enable_ai_features:
            self.ai_workflow = AIWorkflowManager(self)
            asyncio.create_task(self.ai_workflow.start())
            
        logger.info("🧟‍♂️ FrankensteinDB initialized")
        
        # Register MQTT handlers
        
    def _init_evolution_db(self):
        """Initialize time-series evolution database schema"""
        def _init():
            with sqlite3.connect(self.evolution_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS website_evolution (
                        domain TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        structure_hash TEXT NOT NULL,
                        framework_flags INTEGER NOT NULL DEFAULT 0,
                        dna BLOB NOT NULL,
                        PRIMARY KEY (domain, timestamp)
                    )
                """)
                
                # Create indices for efficient querying
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_website_evolution_domain 
                    ON website_evolution(domain)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_website_evolution_timestamp 
                    ON website_evolution(timestamp)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_website_evolution_structure 
                    ON website_evolution(structure_hash)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_website_evolution_frameworks
                    ON website_evolution(framework_flags)
                """)
        
        asyncio.get_event_loop().run_in_executor(self.db_executor, _init)
        
    def _init_search_db(self):
        """Initialize search database schema with FTS5 support"""
        def _init():
            with sqlite3.connect(self.search_db_path) as conn:
                # Create websites FTS5 table
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS websites_fts USING fts5(
                        domain,
                        content,
                        keywords,
                        framework_list,
                        page_type,
                        timestamp UNINDEXED,
                        structure_hash UNINDEXED,
                        dna_json UNINDEXED
                    )
                """)
                
                # Create metadata table for non-searchable data
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS website_metadata (
                        domain TEXT PRIMARY KEY,
                        last_update INTEGER NOT NULL,
                        framework_flags INTEGER NOT NULL DEFAULT 0,
                        accessibility_score REAL NOT NULL DEFAULT 0.0,
                        success_rate REAL NOT NULL DEFAULT 0.0,
                        avg_load_time INTEGER NOT NULL DEFAULT 3000
                    )
                """)
                
                # Create framework statistics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS framework_stats (
                        framework TEXT PRIMARY KEY,
                        usage_count INTEGER NOT NULL DEFAULT 0,
                        last_seen INTEGER NOT NULL,
                        avg_accessibility REAL NOT NULL DEFAULT 0.0
                    )
                """)
        
        asyncio.get_event_loop().run_in_executor(self.search_executor, _init)
        
    async def search_websites(self, query: str, filters: Optional[Dict] = None, 
                            limit: int = 20) -> List[Dict]:
        """
        Search websites by content and metadata
        
        Args:
            query: Search query string
            filters: Optional filters (framework, accessibility_min, page_type)
            limit: Maximum number of results
            
        Returns:
            List of search results with relevance scores
        """
        def _search():
            with sqlite3.connect(self.search_db_path) as conn:
                # Build query conditions
                conditions = []
                params = []
                
                # Add content match
                conditions.append("websites_fts MATCH ?")
                params.append(query)
                
                # Add filters
                if filters:
                    if 'framework' in filters:
                        conditions.append("framework_list LIKE ?")
                        params.append(f"%{filters['framework']}%")
                    
                    if 'page_type' in filters:
                        conditions.append("page_type = ?")
                        params.append(filters['page_type'])
                    
                    if 'accessibility_min' in filters:
                        conditions.append("""
                            domain IN (
                                SELECT domain FROM website_metadata 
                                WHERE accessibility_score >= ?
                            )
                        """)
                        params.append(filters['accessibility_min'])
                
                # Execute search
                query_sql = f"""
                    SELECT 
                        websites_fts.domain,
                        websites_fts.content,
                        websites_fts.framework_list,
                        websites_fts.page_type,
                        websites_fts.timestamp,
                        website_metadata.accessibility_score,
                        website_metadata.success_rate,
                        website_metadata.avg_load_time,
                        bm25(websites_fts) as relevance
                    FROM websites_fts
                    LEFT JOIN website_metadata 
                        ON websites_fts.domain = website_metadata.domain
                    WHERE {' AND '.join(conditions)}
                    ORDER BY relevance DESC
                    LIMIT ?
                """
                params.append(limit)
                
                results = []
                for row in conn.execute(query_sql, params):
                    results.append({
                        'domain': row[0],
                        'content_preview': row[1][:200] + '...',
                        'frameworks': row[2].split(','),
                        'page_type': row[3],
                        'timestamp': row[4],
                        'accessibility_score': row[5],
                        'success_rate': row[6],
                        'avg_load_time_ms': row[7],
                        'relevance_score': row[8]
                    })
                
                return results
        
        return await asyncio.get_event_loop().run_in_executor(
            self.search_executor, _search
        )
    
    async def index_website(self, dna: WebsiteDNA, content_description: str = "",
                          keywords: List[str] = None):
        """
        Add or update website in search index
        
        Args:
            dna: WebsiteDNA instance
            content_description: Extracted content description
            keywords: Additional keywords for searchability
        """
        def _index():
            with sqlite3.connect(self.search_db_path) as conn:
                # Delete existing entry if any
                conn.execute("DELETE FROM websites_fts WHERE domain = ?", 
                           (dna.domain,))
                
                # Index new content
                frameworks = decode_frameworks(dna.framework_flags)
                conn.execute("""
                    INSERT INTO websites_fts (
                        domain, content, keywords, framework_list,
                        page_type, timestamp, structure_hash, dna_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dna.domain,
                    content_description,
                    ','.join(keywords or []),
                    ','.join(frameworks),
                    dna.page_type,
                    int(dna.timestamp),
                    dna.structure_hash,
                    json.dumps(dna.get_analytics())
                ))
                
                # Update metadata
                conn.execute("""
                    INSERT OR REPLACE INTO website_metadata (
                        domain, last_update, framework_flags,
                        accessibility_score, success_rate, avg_load_time
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    dna.domain,
                    int(dna.timestamp),
                    dna.framework_flags,
                    dna.accessibility_score,
                    dna.success_rate,
                    dna.avg_load_time_ms
                ))
                
                # Update framework statistics
                timestamp = int(time.time())
                for framework in frameworks:
                    conn.execute("""
                        INSERT INTO framework_stats (
                            framework, usage_count, last_seen, avg_accessibility
                        ) VALUES (?, 1, ?, ?)
                        ON CONFLICT(framework) DO UPDATE SET
                            usage_count = usage_count + 1,
                            last_seen = ?,
                            avg_accessibility = (
                                avg_accessibility * usage_count + ?
                            ) / (usage_count + 1)
                    """, (
                        framework, timestamp, dna.accessibility_score,
                        timestamp, dna.accessibility_score
                    ))
                
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(self.search_executor, _index)
        
    async def store_evolution(self, dna: WebsiteDNA):
        """
        Store website DNA with time-series partitioning
        
        Args:
            dna: WebsiteDNA instance to store
        """
        def _store():
            with sqlite3.connect(self.evolution_db_path) as conn:
                conn.execute("""
                    INSERT INTO website_evolution (
                        domain, timestamp, structure_hash, 
                        framework_flags, dna
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    dna.domain,
                    int(dna.timestamp),
                    dna.structure_hash,
                    dna.framework_flags,
                    dna.compress()
                ))
                conn.commit()
        
        await asyncio.get_event_loop().run_in_executor(self.db_executor, _store)

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
            cutoff = int(time.time() - (hours * 3600))
            with sqlite3.connect(self.evolution_db_path) as conn:
                results = []
                for row in conn.execute("""
                    SELECT dna FROM website_evolution
                    WHERE domain = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                """, (domain, cutoff)):
                    results.append(WebsiteDNA.decompress(row[0]))
                return results
        
        return await asyncio.get_event_loop().run_in_executor(self.db_executor, _query)

    async def find_similar_structures(self, structure_hash: str, 
                                   limit: int = 10) -> List[Tuple[str, float]]:
        """
        Find websites with similar structural hashes
        
        Args:
            structure_hash: Reference structure hash
            limit: Maximum number of results
            
        Returns:
            List of (domain, similarity_score) tuples
        """
        def _query():
            with sqlite3.connect(self.evolution_db_path) as conn:
                # Get latest entries for each domain
                results = conn.execute("""
                    WITH latest_entries AS (
                        SELECT domain, structure_hash,
                            ROW_NUMBER() OVER (
                                PARTITION BY domain 
                                ORDER BY timestamp DESC
                            ) as rn
                        FROM website_evolution
                    )
                    SELECT domain, structure_hash
                    FROM latest_entries
                    WHERE rn = 1
                """).fetchall()
                
                # Calculate similarities (simplified Hamming distance for demo)
                similarities = []
                for domain, hash_value in results:
                    if hash_value != structure_hash:  # Skip exact matches
                        similarity = sum(a == b for a, b in zip(hash_value, structure_hash))
                        similarities.append((domain, similarity / len(structure_hash)))
                
                # Return top matches
                return sorted(similarities, key=lambda x: x[1], reverse=True)[:limit]
        
        return await asyncio.get_event_loop().run_in_executor(self.db_executor, _query)

    async def get_framework_evolution(self, framework_flag: int, 
                                   limit: int = 1000) -> List[Dict]:
        """
        Track framework adoption over time
        
        Args:
            framework_flag: Bit flag for specific framework
            limit: Maximum number of results
            
        Returns:
            List of dictionaries with domain, timestamp, and framework flags
        """
        def _query():
            with sqlite3.connect(self.evolution_db_path) as conn:
                results = []
                for row in conn.execute("""
                    SELECT domain, timestamp, framework_flags
                    FROM website_evolution
                    WHERE framework_flags & ? != 0
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (framework_flag, limit)):
                    results.append({
                        'domain': row[0],
                        'timestamp': row[1],
                        'frameworks': decode_frameworks(row[2])
                    })
                return results
        
        return await asyncio.get_event_loop().run_in_executor(self.db_executor, _query)

    async def load_scraping_config(self, website: str) -> Optional[ScrapingConfig]:
        """
        Load scraping configuration from YAML and cache in Redis
        
        Args:
            website: Website domain (e.g., 'example.com')
            
        Returns:
            ScrapingConfig if found, None otherwise
        """
        # First try Redis cache
        config = await self.context_store.get_scraping_logic(website)
        if config:
            return ScrapingConfig(**config)

        # Try loading from YAML file
        config_path = Path(f"scraping_configs/{website}.yml")
        if not config_path.exists():
            logger.warning(f"No configuration found for {website}")
            return None

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
                config = ScrapingConfig(**data)
                
                # Cache in Redis
                await self.context_store.store_scraping_logic(
                    website, config.to_dict(), ttl=86400
                )
                return config
                
        except Exception as e:
            logger.error(f"Error loading config for {website}: {e}")
            return None

    async def save_scraping_config(self, config: ScrapingConfig) -> bool:
        """
        Save scraping configuration to YAML and Redis
        
        Args:
            config: Complete scraping configuration
            
        Returns:
            True if saved successfully
        """
        try:
            # Update timestamp
            config.last_updated = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Save to YAML
            config_path = Path(f"scraping_configs/{config.website}.yml")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config.to_dict(), f, sort_keys=False)
            
            # Cache in Redis
            await self.context_store.store_scraping_logic(
                config.website, config.to_dict(), ttl=86400
            )
            
            logger.info(f"Saved config for {config.website} v{config.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config for {config.website}: {e}")
            return False

    async def update_scraping_config(self, website: str, 
                                   new_logic: Dict[str, Any],
                                   change_reason: str,
                                   ai_generated: bool = False) -> Optional[ScrapingConfig]:
        """
        Update scraping configuration with version control
        
        Args:
            website: Website domain
            new_logic: Updated scraping logic
            change_reason: Reason for the change
            ai_generated: Whether changes were AI-generated
            
        Returns:
            Updated ScrapingConfig or None on failure
        """
        # Load current config
        config = await self.load_scraping_config(website)
        if not config:
            return None
            
        # Determine version bump
        old_version = tuple(map(int, config.version.split('.')))
        if ai_generated:
            # AI changes bump patch version
            new_version = (old_version[0], old_version[1], old_version[2] + 1)
        else:
            # Manual changes bump minor version
            new_version = (old_version[0], old_version[1] + 1, 0)
            
        version_str = '.'.join(map(str, new_version))
        
        # Update config
        config.version = version_str
        config.scraping_logic = new_logic
        
        # Add change history
        if 'adaptation_history' not in config.ai_metadata:
            config.ai_metadata['adaptation_history'] = []
            
        config.ai_metadata['adaptation_history'].append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'version': version_str,
            'reason': change_reason,
            'ai_generated': ai_generated
        })
        
        # Save updated config
        if await self.save_scraping_config(config):
            return config
            
        return None

    async def list_all_configs(self) -> List[str]:
        """
        List all available scraping configurations
        
        Returns:
            List of website domains with configurations
        """
        try:
            config_path = Path("scraping_configs")
            if not config_path.exists():
                return []
                
            return [
                p.stem for p in config_path.glob("*.yml")
                if p.is_file()
            ]
            
        except Exception as e:
            logger.error(f"Error listing configs: {e}")
            return []

    async def validate_config(self, config: ScrapingConfig) -> List[str]:
        """
        Validate scraping configuration
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Basic validation
        if not config.website:
            errors.append("Missing website domain")
            
        if not config.version:
            errors.append("Missing version number")
            
        if not isinstance(config.scraping_logic, dict):
            errors.append("Invalid scraping logic format")
            
        # Check required scraping logic fields
        required_fields = ['selectors', 'patterns', 'execution']
        for field in required_fields:
            if field not in config.scraping_logic:
                errors.append(f"Missing required field: {field}")
                
        # Validate selectors
        selectors = config.scraping_logic.get('selectors', {})
        if not isinstance(selectors, dict):
            errors.append("Invalid selectors format")
        else:
            for name, selector in selectors.items():
                if not isinstance(selector, str):
                    errors.append(f"Invalid selector format: {name}")
                    
        # Validate execution steps
        execution = config.scraping_logic.get('execution', [])
        if not isinstance(execution, list):
            errors.append("Invalid execution steps format")
        else:
            for i, step in enumerate(execution):
                if not isinstance(step, dict):
                    errors.append(f"Invalid step format at index {i}")
                if 'action' not in step:
                    errors.append(f"Missing action in step {i}")
                    
        return errors
        self.mqtt.register_handler('db/+/queries', self.mqtt_handlers.handle_db_query)
        self.mqtt.register_handler('scraper/+/results', self.mqtt_handlers.handle_scraper_result)
        self.mqtt.register_handler('db/ai/task', self._handle_ai_task)
        self.mqtt.register_handler('scraper/+/context', self._handle_scraper_context)
        
        # Initialize connections
        asyncio.create_task(self.dna_store.connect())
        self.mqtt.start()
        
        # Start status updates
        if self.enable_ai_features:
            asyncio.create_task(self.mqtt_handlers.publish_db_status())

        logger.info("🧟‍♂️ FrankensteinDB monster is ALIVE!")
        if enable_ai_features:
            logger.info("🤖 AI features enabled - multi-database Redis support active")

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
            self.search_index.index_website(dna, content_description, keywords or []),
            
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
        trends = self._analyze_evolution_trends(evolution) if isinstance(evolution, list) else {}
        
        return {
            'domain': domain,
            'statistics': stats,
            'evolution_timeline': evolution[-10:] if isinstance(evolution, list) and evolution else [],  # Last 10 snapshots
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
        if not isinstance(evolution, list) or len(evolution) < 2:
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
    
    async def _handle_ai_task(self, topic: str, payload: Dict):
        """Handle incoming AI task requests"""
        if not self.enable_ai_features:
            return
            
        try:
            task_type = payload.get('type')
            if task_type in ['clean', 'structure', 'solve']:
                await self.ai_workflow.queue_task(payload)
        except Exception as e:
            logger.error(f"Error handling AI task: {str(e)}")

    async def store_ai_knowledge(self, knowledge_id: str, knowledge_data: Dict, ttl: int = 86400):
        """
        Store AI-learned knowledge and patterns

        Args:
            knowledge_id: Unique knowledge identifier
            knowledge_data: Knowledge data (patterns, rules, insights)
            ttl: Time to live in seconds
        """
        if not self.enable_ai_features:
            logger.warning("AI features disabled")
            return

        await self.context_store.store_ai_knowledge(knowledge_id, knowledge_data, ttl)
        logger.info(f"🧠 Stored AI knowledge: {knowledge_id}")

    async def get_ai_knowledge(self, knowledge_id: str) -> Optional[Dict]:
        """
        Retrieve AI knowledge

        Args:
            knowledge_id: Knowledge identifier

        Returns:
            Knowledge data or None
        """
        if not self.enable_ai_features:
            return None

        return await self.context_store.get_ai_knowledge(knowledge_id)

    async def store_scraping_logic(self, site_pattern: str, scraping_config: Dict, ttl: int = 604800):
        """
        Store scraping patterns and logic for websites

        Args:
            site_pattern: Site pattern (domain or URL pattern)
            scraping_config: Scraping configuration
            ttl: Time to live in seconds
        """
        if not self.enable_ai_features:
            logger.warning("AI features disabled")
            return

        await self.context_store.store_scraping_logic(site_pattern, scraping_config, ttl)
        logger.info(f"🕷️ Stored scraping logic for: {site_pattern}")

    async def get_scraping_logic(self, site_pattern: str) -> Optional[Dict]:
        """
        Retrieve scraping logic for a site

        Args:
            site_pattern: Site pattern

        Returns:
            Scraping configuration or None
        """
        if not self.enable_ai_features:
            return None

        return await self.context_store.get_scraping_logic(site_pattern)

    async def cache_website_dna(self, url: str, dna_data: Dict, ttl: int = 3600):
        """
        Cache website DNA analysis results

        Args:
            url: Website URL
            dna_data: DNA analysis data
            ttl: Time to live in seconds
        """
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()

        if self.enable_ai_features:
            await self.context_store.cache_website_dna(url_hash, dna_data, ttl)

        logger.info(f"🧬 Cached DNA for: {url[:50]}...")

    async def get_cached_dna(self, url: str) -> Optional[Dict]:
        """
        Retrieve cached website DNA

        Args:
            url: Website URL

        Returns:
            DNA data or None
        """
        if not self.enable_ai_features:
            return None

        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return await self.context_store.get_cached_dna(url_hash)

    async def store_session_analysis(self, session_id: str, analysis_data: Dict, ttl: int = 7200):
        """
        Store session analysis and AI insights

        Args:
            session_id: Session identifier
            analysis_data: Analysis results and insights
            ttl: Time to live in seconds
        """
        if not self.enable_ai_features:
            logger.warning("AI features disabled")
            return

        await self.context_store.store_session_metadata(session_id, analysis_data, ttl)
        logger.info(f"📊 Stored session analysis: {session_id}")

    async def get_session_analysis(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve session analysis data

        Args:
            session_id: Session identifier

        Returns:
            Analysis data or None
        """
        if not self.enable_ai_features:
            return None

        return await self.context_store.get_session_metadata(session_id)

    def get_ai_system_stats(self) -> Dict:
        """
        Get comprehensive AI system statistics

        Returns:
            Dictionary with AI system stats
        """
        base_stats = self.context_store.get_enhanced_stats()
        base_stats.update({
            'ai_features_enabled': self.enable_ai_features,
            'components': {
                'evolution_db': 'active',
                'search_index': 'active',
                'context_store': 'active',
                'blob_storage': 'active',
                'ai_enhancements': 'active' if self.enable_ai_features else 'disabled'
            }
        })
        return base_stats

    async def close(self):
        """Clean up all resources"""
        logger.info("🧟‍♂️ FrankensteinDB monster going to sleep...")

        # Close all components
        self.dna_store.close()
        self.search_index.close()
        self.context_store.close()

        # Blob storage doesn't need explicit closing

        logger.info("💤 FrankensteinDB resources cleaned up")