"""
MQTT Core - Consolidated MQTT client with DNA-aware routing

Provides secure MQTT communication with DNA-based pattern matching,
intelligent message routing, and comprehensive error handling.
"""

import asyncio
import json
import logging
import ssl
import time
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import paho.mqtt.client as mqtt

from .website_dna import WebsiteDNA

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Component types for MQTT clients"""
    STORAGE = "storage"
    SCRAPER = "scraper"
    AI = "ai"
    ORCHESTRATOR = "orchestrator"
    MONITOR = "monitor"


class MessagePriority(Enum):
    """Message priority levels"""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class TopicPattern:
    """DNA-aware topic pattern matching"""
    def __init__(self, pattern: str, handler: Callable, priority: MessagePriority = MessagePriority.NORMAL):
        self.pattern = pattern
        self.handler = handler
        self.priority = priority
        self.dna_filters: List[Dict] = []
        
    def add_dna_filter(self, field: str, value: Any, match_type: str = "exact"):
        """Add DNA-based message filter"""
        self.dna_filters.append({
            "field": field,
            "value": value,
            "match_type": match_type
        })
        
    def matches_dna(self, dna: Optional[WebsiteDNA]) -> bool:
        """Check if message matches DNA filters"""
        if not dna or not self.dna_filters:
            return True
            
        for filter in self.dna_filters:
            field_value = getattr(dna, filter["field"], None)
            if not field_value:
                continue
                
            if filter["match_type"] == "exact":
                if field_value != filter["value"]:
                    return False
            elif filter["match_type"] == "contains":
                if filter["value"] not in field_value:
                    return False
            elif filter["match_type"] == "threshold":
                if float(field_value) < float(filter["value"]):
                    return False
                    
        return True


class MQTTCore:
    """
    Core MQTT client with advanced features
    
    Provides secure communication, DNA-based routing,
    automatic reconnection, and comprehensive error handling.
    """
    
    def __init__(self,
                 client_id: str,
                 host: str = "localhost",
                 port: int = 1883,
                 component_type: ComponentType = ComponentType.MONITOR,
                 use_ssl: bool = False,
                 ca_certs: Optional[str] = None):
                 
        self.client_id = client_id
        self.host = host
        self.port = port
        self.component_type = component_type
        self.use_ssl = use_ssl
        self.ca_certs = ca_certs
        
        # Connection state
        self.connected = False
        self.connecting = False
        self.last_connection = 0
        self.reconnect_interval = 1.0
        
        # Message handling
        self.topic_patterns: Dict[str, TopicPattern] = {}
        self.message_queue = asyncio.Queue()
        self.processing_task = None
        
        # Create MQTT client
        self.client = mqtt.Client(client_id)
        self._setup_client()
        
    def _setup_client(self):
        """Configure MQTT client"""
        if self.use_ssl:
            self.client.tls_set(
                ca_certs=self.ca_certs,
                cert_reqs=ssl.CERT_REQUIRED
            )
            
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_publish = self._on_publish
        
        # Enable logging
        self.client.enable_logger(logger)
        
    def _on_connect(self, client, userdata, flags, rc):
        """Handle connection establishment"""
        if rc == 0:
            self.connected = True
            self.connecting = False
            self.last_connection = time.time()
            self.reconnect_interval = 1.0
            
            # Resubscribe to topics
            for pattern in self.topic_patterns.values():
                self.client.subscribe(pattern.pattern)
                
            logger.info(f"Connected to MQTT broker at {self.host}:{self.port}")
        else:
            logger.error(f"Connection failed with code {rc}")
            
    def _on_disconnect(self, client, userdata, rc):
        """Handle disconnection"""
        self.connected = False
        if rc != 0:
            logger.warning("Unexpected disconnection, will reconnect...")
            asyncio.create_task(self._reconnect())
            
    def _on_message(self, client, userdata, message):
        """Handle incoming messages"""
        try:
            payload = json.loads(message.payload)
            
            # Add to processing queue
            asyncio.create_task(self.message_queue.put({
                "topic": message.topic,
                "payload": payload,
                "qos": message.qos,
                "timestamp": time.time()
            }))
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON payload on topic {message.topic}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
    def _on_publish(self, client, userdata, mid):
        """Handle message publication confirmation"""
        logger.debug(f"Message {mid} published successfully")
        
    async def _reconnect(self):
        """Handle reconnection with backoff"""
        if self.connecting:
            return
            
        self.connecting = True
        while not self.connected:
            try:
                self.client.reconnect()
                await asyncio.sleep(self.reconnect_interval)
                self.reconnect_interval = min(self.reconnect_interval * 2, 60)
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                
        self.connecting = False
        
    async def _process_messages(self):
        """Process messages from queue"""
        while True:
            try:
                message = await self.message_queue.get()
                
                # Find matching patterns
                matches = []
                for pattern in self.topic_patterns.values():
                    if mqtt.topic_matches_sub(pattern.pattern, message["topic"]):
                        matches.append(pattern)
                        
                # Sort by priority
                matches.sort(key=lambda p: p.priority.value)
                
                # Process matches
                for pattern in matches:
                    try:
                        await pattern.handler(
                            message["topic"],
                            message["payload"]
                        )
                    except Exception as e:
                        logger.error(f"Handler error for {message['topic']}: {e}")
                        
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await asyncio.sleep(1)
                
    async def start(self):
        """Start MQTT client"""
        try:
            # Connect to broker
            self.client.connect_async(self.host, self.port)
            self.client.loop_start()
            
            # Start message processor
            self.processing_task = asyncio.create_task(
                self._process_messages()
            )
            
            logger.info("MQTT client started")
            
        except Exception as e:
            logger.error(f"Failed to start MQTT client: {e}")
            
    async def stop(self):
        """Stop MQTT client"""
        try:
            # Stop message processing
            if self.processing_task:
                self.processing_task.cancel()
                
            # Disconnect client
            self.client.disconnect()
            self.client.loop_stop()
            
            logger.info("MQTT client stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MQTT client: {e}")
            
    def subscribe(self, topic: str, handler: Callable,
                priority: MessagePriority = MessagePriority.NORMAL) -> TopicPattern:
        """Subscribe to topic with DNA-aware pattern"""
        pattern = TopicPattern(topic, handler, priority)
        self.topic_patterns[topic] = pattern
        
        if self.connected:
            self.client.subscribe(topic)
            
        return pattern
        
    async def publish(self, topic: str, payload: Dict,
                   qos: int = 0, retain: bool = False) -> bool:
        """Publish message to topic"""
        try:
            # Encode payload
            message = json.dumps(payload)
            
            # Publish with QoS
            result = self.client.publish(
                topic,
                message,
                qos=qos,
                retain=retain
            )
            
            return result.rc == mqtt.MQTT_ERR_SUCCESS
            
        except Exception as e:
            logger.error(f"Error publishing to {topic}: {e}")
            return False
            
    async def unsubscribe(self, topic: str):
        """Unsubscribe from topic"""
        try:
            self.client.unsubscribe(topic)
            self.topic_patterns.pop(topic, None)
            
        except Exception as e:
            logger.error(f"Error unsubscribing from {topic}: {e}")
            
    def add_dna_filter(self, topic: str, field: str,
                    value: Any, match_type: str = "exact"):
        """Add DNA filter to topic pattern"""
        if topic in self.topic_patterns:
            self.topic_patterns[topic].add_dna_filter(
                field, value, match_type
            )