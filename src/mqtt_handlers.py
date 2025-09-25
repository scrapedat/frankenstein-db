import time
import asyncio
import logging
from typing import Dict, Optional, List
from .website_dna import WebsiteDNA, decode_frameworks

logger = logging.getLogger(__name__)

class MQTTHandlers:
    """MQTT message handlers for FrankensteinDB"""
    
    def __init__(self, db):
        """Initialize handlers with database instance"""
        self.db = db
    
    async def _handle_db_query(self, topic: str, payload: Dict):
        """
        Handle incoming database queries via MQTT
        
        Args:
            topic: MQTT topic
            payload: Query payload
        """
        try:
            query_type = payload.get('type')
            query_data = payload.get('data', {})
            user_id = topic.split('/')[1]
            
            response = None
            
            if query_type == 'evolution':
                response = await self.query_website_evolution(
                    query_data.get('domain'),
                    query_data.get('hours', 24)
                )
            elif query_type == 'similarity':
                response = await self.find_similar_websites(
                    query_data.get('url'),
                    query_data.get('limit', 10)
                )
            elif query_type == 'intelligence':
                response = await self.get_domain_intelligence(
                    query_data.get('domain')
                )
                
            if response:
                await self.mqtt.publish(
                    f'db/{user_id}/response',
                    {
                        'query_id': payload.get('query_id'),
                        'type': query_type,
                        'data': response
                    }
                )
                
        except Exception as e:
            logger.error(f"Error handling DB query: {str(e)}")

    async def _handle_scraper_result(self, topic: str, payload: Dict):
        """
        Handle incoming scraper results via MQTT
        
        Args:
            topic: MQTT topic
            payload: Scraper result payload
        """
        transaction_started = False
        try:
            scraper_id = topic.split('/')[1]
            result_type = payload.get('type')
            result_data = payload.get('data', {})
            
            if result_type == 'website_snapshot':
                # Validate required fields
                url = result_data.get('url')
                html = result_data.get('html')
                structure = result_data.get('structure')
                
                if not all([url, html, structure]):
                    raise ValueError("Missing required fields in website snapshot")
                
                # Start transaction
                transaction_started = True
                await self.dna_store.start_transaction()
                
                try:
                    # Store with timeout
                    async with asyncio.timeout(30):  # 30 second timeout
                        dna = await self.store_website_snapshot(
                            url=url,
                            html_content=html,
                            structure_fingerprint=structure,
                            user_context=scraper_id
                        )
                except asyncio.TimeoutError:
                    logger.error("Timeout storing website snapshot")
                    if transaction_started:
                        await self.dna_store.rollback_transaction()
                    await self._report_error('store_timeout', scraper_id, url)
                    return
                
                # Analyze for relevant patterns
                if self.enable_ai_features and dna:
                    try:
                        async with asyncio.timeout(15):  # 15 second timeout for analysis
                            insights = await self._analyze_snapshot_for_insights(dna)
                            if insights:
                                await self.mqtt.publish(
                                    f'db/insights/{scraper_id}',
                                    {
                                        'url': url,
                                        'insights': insights
                                    }
                                )
                    except asyncio.TimeoutError:
                        logger.warning("Timeout during insight analysis - continuing without insights")
                        # Don't roll back the transaction - storage succeeded
                
                # Commit transaction
                if transaction_started:
                    await self.dna_store.commit_transaction()
                        
        except ValueError as e:
            logger.error(f"Validation error in scraper result: {str(e)}")
            await self._report_error('validation_error', scraper_id, str(e))
        except Exception as e:
            logger.error(f"Error handling scraper result: {str(e)}")
            if transaction_started:
                await self.dna_store.rollback_transaction()
            await self._report_error('processing_error', scraper_id, str(e))
            
    async def _report_error(self, error_type: str, scraper_id: str, details: str):
        """Report error via MQTT"""
        try:
            await self.mqtt.publish(
                f'db/errors',
                {
                    'type': error_type,
                    'scraper_id': scraper_id,
                    'details': details,
                    'timestamp': time.time()
                }
            )
        except Exception as e:
            logger.error(f"Failed to report error: {str(e)}")

    async def _analyze_snapshot_for_insights(self, dna: WebsiteDNA) -> Optional[Dict]:
        """
        Analyze website snapshot for valuable insights
        
        Args:
            dna: WebsiteDNA instance
            
        Returns:
            Dictionary of insights or None
        """
        insights = {}
        
        # Check for significant structure changes
        evolution = await self.query_website_evolution(dna.domain, hours=24)
        if evolution and len(evolution) > 1:
            prev_hash = evolution[-2]['structure_hash']
            if prev_hash != dna.structure_hash:
                insights['structure_change'] = True
                
        # Check for new frameworks
        if dna.framework_flags:
            frameworks = decode_frameworks(dna.framework_flags)
            insights['frameworks'] = frameworks
            
        # Check accessibility changes
        if evolution and len(evolution) > 1:
            prev_score = evolution[-2]['accessibility_score']
            if abs(dna.accessibility_score - prev_score) > 0.1:
                insights['accessibility_change'] = {
                    'previous': prev_score,
                    'current': dna.accessibility_score
                }
                
        return insights if insights else None

    async def publish_db_status(self):
        """Publish database status updates via MQTT"""
        while True:
            try:
                health = await self.get_system_health()
                await self.mqtt.publish(
                    'db/status',
                    {
                        'timestamp': time.time(),
                        'health': health
                    }
                )
            except Exception as e:
                logger.error(f"Error publishing status: {str(e)}")
                
            await asyncio.sleep(60)  # Update every minute