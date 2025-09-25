"""
Context Update System for FrankensteinDB

Manages targeted context updates from DB AI to scraper AI,
ensuring relevant information is shared efficiently.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set
import json

logger = logging.getLogger(__name__)

class ContextUpdateManager:
    """
    Manages the flow of context updates between DB AI and scraper AI
    """
    
    def __init__(self, db, relevance_threshold: float = 0.7):
        """
        Initialize context update manager
        
        Args:
            db: FrankensteinDB instance
            relevance_threshold: Minimum relevance score for updates (0-1)
        """
        self.db = db
        self.relevance_threshold = relevance_threshold
        self.active_contexts: Dict[str, Dict] = {}
        self.update_cache: Dict[str, List[Dict]] = {}
        
    async def register_scraper_context(self, scraper_id: str, context: Dict):
        """
        Register a scraper's current context
        
        Args:
            scraper_id: Unique scraper identifier
            context: Current scraping context
        """
        self.active_contexts[scraper_id] = {
            'context': context,
            'timestamp': time.time(),
            'interests': context.get('interests', []),
            'domains': context.get('domains', []),
            'patterns': context.get('patterns', [])
        }
        
        # Process any cached updates
        if scraper_id in self.update_cache:
            await self._process_cached_updates(scraper_id)
            
    async def publish_insight(self, insight: Dict):
        """
        Publish a new insight from DB AI
        
        Args:
            insight: Discovered insight
        """
        for scraper_id, context in self.active_contexts.items():
            relevance = await self._calculate_relevance(insight, context)
            
            if relevance >= self.relevance_threshold:
                await self._send_update(scraper_id, insight, relevance)
            elif relevance > 0.3:  # Cache somewhat relevant updates
                self._cache_update(scraper_id, insight)
                
    async def _calculate_relevance(self, insight: Dict, context: Dict) -> float:
        """
        Calculate relevance score of insight to scraper context
        
        Args:
            insight: Discovered insight
            context: Scraper context
            
        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        matches = 0
        
        # Check domain relevance
        if insight.get('domain') in context['domains']:
            score += 0.4
            matches += 1
            
        # Check pattern relevance
        insight_patterns = insight.get('patterns', [])
        context_patterns = context['patterns']
        pattern_matches = len(set(insight_patterns) & set(context_patterns))
        if pattern_matches:
            score += 0.3 * (pattern_matches / len(context_patterns))
            matches += 1
            
        # Check interest relevance
        insight_topics = insight.get('topics', [])
        context_interests = context['interests']
        interest_matches = len(set(insight_topics) & set(context_interests))
        if interest_matches:
            score += 0.3 * (interest_matches / len(context_interests))
            matches += 1
            
        # Normalize score based on matches
        return score / max(1, matches)
        
    async def _send_update(self, scraper_id: str, insight: Dict, relevance: float):
        """
        Send context update to scraper
        
        Args:
            scraper_id: Target scraper
            insight: Update insight
            relevance: Calculated relevance score
        """
        try:
            await self.db.mqtt.publish(
                f'db/context/{scraper_id}',
                {
                    'type': 'context_update',
                    'timestamp': time.time(),
                    'insight': insight,
                    'relevance': relevance,
                    'source': 'db_ai'
                }
            )
            logger.debug(f"📤 Sent context update to {scraper_id}")
            
        except Exception as e:
            logger.error(f"Error sending context update: {str(e)}")
            
    def _cache_update(self, scraper_id: str, insight: Dict):
        """
        Cache update for future processing
        
        Args:
            scraper_id: Target scraper
            insight: Update insight
        """
        if scraper_id not in self.update_cache:
            self.update_cache[scraper_id] = []
            
        self.update_cache[scraper_id].append({
            'insight': insight,
            'timestamp': time.time()
        })
        
        # Limit cache size
        self.update_cache[scraper_id] = self.update_cache[scraper_id][-100:]
        
    async def _process_cached_updates(self, scraper_id: str):
        """
        Process cached updates for scraper
        
        Args:
            scraper_id: Target scraper
        """
        if scraper_id not in self.update_cache:
            return
            
        context = self.active_contexts.get(scraper_id)
        if not context:
            return
            
        current_time = time.time()
        
        # Process recent cached updates
        for update in self.update_cache[scraper_id]:
            # Skip old updates
            if current_time - update['timestamp'] > 3600:  # 1 hour
                continue
                
            relevance = await self._calculate_relevance(
                update['insight'],
                context
            )
            
            if relevance >= self.relevance_threshold:
                await self._send_update(
                    scraper_id,
                    update['insight'],
                    relevance
                )
                
        # Clear processed cache
        self.update_cache[scraper_id] = []
        
    async def cleanup_old_contexts(self):
        """Remove old contexts periodically"""
        while True:
            try:
                current_time = time.time()
                to_remove = []
                
                for scraper_id, context in self.active_contexts.items():
                    if current_time - context['timestamp'] > 3600:  # 1 hour
                        to_remove.append(scraper_id)
                        
                for scraper_id in to_remove:
                    del self.active_contexts[scraper_id]
                    if scraper_id in self.update_cache:
                        del self.update_cache[scraper_id]
                        
            except Exception as e:
                logger.error(f"Error cleaning contexts: {str(e)}")
                
            await asyncio.sleep(300)  # Check every 5 minutes