"""
Selective Context Consumer for Scraper AI

Enables the scraper AI to intelligently filter and consume
context updates from the database AI based on relevance
and current task priorities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Tuple, TypeVar
from dataclasses import dataclass

from .website_dna import (
    WebsiteDNA, 
    calculate_accessibility_score,
    decode_frameworks
)
from .ai_workflow import (
    AIWorkflowManager,
    AIWorkflowState,
    DatabaseTaskType
)

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class UpdateContext:
    """Context for batched updates"""
    updates: List[Dict]
    priority: float
    timestamp: float
    dna_signature: Optional[str] = None

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable
import json

logger = logging.getLogger(__name__)

class SelectiveContextConsumer:
    """
    Manages selective consumption of context updates for scraper AI
    """
    
    def __init__(self,
                mqtt_client,
                scraper_id: str,
                ai_workflow: AIWorkflowManager,
                relevance_threshold: float = 0.5,
                update_callback: Optional[Callable] = None,
                batch_size: int = 10,
                batch_timeout: float = 5.0):
        """
        Initialize context consumer
        
        Args:
            mqtt_client: MQTT client instance
            scraper_id: Unique scraper identifier
            relevance_threshold: Minimum relevance for immediate processing
            update_callback: Optional callback for relevant updates
        """
        self.mqtt = mqtt_client
        self.scraper_id = scraper_id
        self.ai_workflow = ai_workflow
        self.relevance_threshold = relevance_threshold
        self.update_callback = update_callback
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        self.current_task: Optional[Dict] = None
        self.task_priorities: Dict[str, float] = {}
        self.deferred_updates: List[Dict] = []
        self.context_subscriptions: Dict[str, float] = {}
        self.update_batches: Dict[str, UpdateContext] = {}
        self.dna_patterns: Dict[str, WebsiteDNA] = {}
        
        # Start update processor
        asyncio.create_task(self._process_update_batches())
        
    def set_current_task(self, task: Dict):
        """
        Set current scraping task
        
        Args:
            task: Current task configuration
        """
        self.current_task = task
        self.task_priorities = self._calculate_task_priorities(task)
        
        # Publish context for DB AI
        asyncio.create_task(self._publish_context())
        
    def subscribe_to_context(self, topic: str, priority: float = 1.0):
        """
        Subscribe to specific context updates
        
        Args:
            topic: Context topic (e.g., 'frameworks', 'accessibility')
            priority: Topic priority (0-1)
        """
        self.context_subscriptions[topic] = priority
        
    def unsubscribe_from_context(self, topic: str):
        """
        Unsubscribe from context updates
        
        Args:
            topic: Context topic to unsubscribe from
        """
        self.context_subscriptions.pop(topic, None)
        
    async def handle_update(self, update: Dict):
        """
        Handle incoming context update with DNA-aware batching
        
        Args:
            update: Context update from DB AI
        """
        try:
            relevance, dna_sig = self._calculate_task_relevance(update)
            
            if relevance >= self.relevance_threshold:
                # Add to appropriate batch
                batch_key = f"{dna_sig or 'default'}:{relevance}"
                
                if batch_key not in self.update_batches:
                    self.update_batches[batch_key] = UpdateContext(
                        updates=[],
                        priority=relevance,
                        timestamp=time.time(),
                        dna_signature=dna_sig
                    )
                    
                batch = self.update_batches[batch_key]
                batch.updates.append(update)
                
                # Process batch if full
                if len(batch.updates) >= self.batch_size:
                    await self._process_batch(batch_key)
            else:
                self._defer_update(update)
                
        except Exception as e:
            logger.error(f"Error handling update: {e}")
            
    async def _process_update(self, update: Dict):
        """
        Process relevant context update
        
        Args:
            update: Context update to process
        """
        if self.update_callback:
            await self.update_callback(update)
        else:
            logger.debug(f"📥 Relevant update received: {update['insight']['topic']}")
            
    def _defer_update(self, update: Dict):
        """
        Defer update for later processing
        
        Args:
            update: Context update to defer
        """
        self.deferred_updates.append({
            'update': update,
            'timestamp': time.time()
        })
        
        # Limit deferred updates
        self.deferred_updates = self.deferred_updates[-50:]
        
    def _match_dna_pattern(self, dna: WebsiteDNA, task: Dict) -> bool:
        """
        Check if DNA pattern matches task requirements
        
        Args:
            dna: Website DNA pattern
            task: Task configuration
            
        Returns:
            True if pattern matches task needs
        """
        # Check framework requirements
        if 'target_frameworks' in task:
            frameworks = set(decode_frameworks(dna.framework_flags))
            if not frameworks & set(task['target_frameworks']):
                return False
                
        # Check performance requirements
        if task.get('min_performance'):
            if dna.avg_load_time_ms > task['min_performance']:
                return False
                
        # Check accessibility requirements
        if task.get('min_accessibility'):
            if dna.accessibility_score < task['min_accessibility']:
                return False
                
        # Check error pattern thresholds
        if task.get('max_error_rate'):
            if dna.success_rate < (1 - task['max_error_rate']):
                return False
                
        return True
    
    def _calculate_task_priorities(self, task: Dict) -> Dict[str, float]:
        """
        Calculate priority weights using DNA patterns
        
        Args:
            task: Task configuration
            
        Returns:
            Dictionary of priority weights
        """
        priorities = {}
        
        # Domain priority
        if 'domain' in task:
            priorities['domain'] = 1.0
            
        # Framework detection priority
        if task.get('detect_frameworks', False):
            priorities['frameworks'] = 0.8
            
        # Accessibility priority
        if task.get('check_accessibility', False):
            priorities['accessibility'] = 0.7
            
        # Performance priority
        if task.get('measure_performance', False):
            priorities['performance'] = 0.6
            
        return priorities
        
    def _calculate_task_relevance(self, insight: Dict) -> Tuple[float, Optional[str]]:
        """
        Calculate insight relevance using DNA patterns
        
        Args:
            insight: Context insight
            
        Returns:
            Tuple of (relevance score 0-1, DNA signature if matched)
        """
        if not self.current_task:
            return 0.0, None
            
        score = 0.0
        dna_signature = None
        
        # Check domain relevance
        domain = insight.get('domain')
        if domain and domain in self.dna_patterns:
            dna = self.dna_patterns[domain]
            
            # Match against task DNA patterns
            if self._match_dna_pattern(dna, self.current_task):
                score += 0.4
                dna_signature = dna.structure_hash
        
        # Framework relevance
        if 'frameworks' in insight and 'target_frameworks' in self.current_task:
            common = set(insight['frameworks']) & set(self.current_task['target_frameworks'])
            if common:
                score += 0.3 * (len(common) / len(self.current_task['target_frameworks']))
        
        # Accessibility relevance
        if self.current_task.get('check_accessibility'):
            score += 0.2 * calculate_accessibility_score(insight.get('accessibility_features', {}))
        
        # Performance relevance 
        if self.current_task.get('measure_performance'):
            perf_score = insight.get('performance', {}).get('score', 0)
            score += 0.1 * min(perf_score / 100, 1.0)
            
        return score, dna_signature
        
    async def _publish_context(self):
        """Publish current context to DB AI"""
        try:
            context = {
                'task': self.current_task,
                'priorities': self.task_priorities,
                'subscriptions': self.context_subscriptions,
                'timestamp': time.time()
            }
            
            await self.mqtt.publish(
                f'scraper/{self.scraper_id}/context',
                {
                    'type': 'context_update',
                    'context': context
                }
            )
            
        except Exception as e:
            logger.error(f"Error publishing context: {str(e)}")
    async def _process_batch(self, batch_key: str):
        """
        Process a batch of related updates
        
        Args:
            batch_key: Key identifying the batch
        """
        try:
            batch = self.update_batches.pop(batch_key)
            
            # Group by DNA pattern
            if batch.dna_signature:
                await self._process_dna_batch(batch)
            else:
                # Process individual updates
                for update in batch.updates:
                    await self._process_update(update)
                    
        except Exception as e:
            logger.error(f"Error processing batch {batch_key}: {e}")
            
    async def _process_dna_batch(self, batch: UpdateContext):
        """
        Process updates sharing DNA pattern
        
        Args:
            batch: Batch of related updates
        """
        try:
            # Create optimized task
            task = {
                'type': DatabaseTaskType.DNA_UPDATE,
                'priority': batch.priority * 2,  # Boost priority for DNA matches
                'updates': batch.updates,
                'dna_signature': batch.dna_signature
            }
            
            # Queue task in AI workflow
            await self.ai_workflow.queue_task(task)
            
        except Exception as e:
            logger.error(f"Error processing DNA batch: {e}")
    
    async def _process_update_batches(self):
        """
        Periodically process update batches
        """
        while True:
            try:
                current_time = time.time()
                
                # Process timed-out batches
                for batch_key, batch in list(self.update_batches.items()):
                    if current_time - batch.timestamp >= self.batch_timeout:
                        await self._process_batch(batch_key)
                        
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(5.0)  # Back off on error
            
    async def process_deferred_updates(self):
        """Process deferred updates periodically"""
        while True:
            try:
                current_time = time.time()
                
                # Process updates older than 5 minutes
                for entry in self.deferred_updates[:]:
                    if current_time - entry['timestamp'] > 300:
                        update = entry['update']
                        # Recalculate relevance
                        task_relevance = self._calculate_task_relevance(
                            update.get('insight', {})
                        )
                        
                        if task_relevance >= self.relevance_threshold:
                            await self._process_update(update)
                            
                        self.deferred_updates.remove(entry)
                        
            except Exception as e:
                logger.error(f"Error processing deferred updates: {str(e)}")
                
            await asyncio.sleep(60)  # Check every minute