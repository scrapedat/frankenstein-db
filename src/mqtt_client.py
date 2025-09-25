"""
mqtt_client.py for frankenstein_db AI Communication

Handles real-time communication between the database AI system
and the scraper orchestrator.
"""

import asyncio
import json
import logging
from typing import Dict, Optional, Callable, Any
import paho.mqtt.client as mqtt
from paho.mqtt.client import MQTTMessage
import threading

logger = logging.getLogger(__name__)

class DatabaseMQTTClient:
    """
    MQTT client specialized for database AI communication
    """
    
    def __init__(self, 
                broker_host: str = 'localhost',
                broker_port: int = 1883,
                client_id: str = 'frankenstein_db'):
        """
        Initialize MQTT client
        
        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            client_id: Unique client identifier
        """
        self.client = mqtt.Client(client_id=client_id)
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.handlers = {}
        self.connected = False
        self._setup_client()
        
    def _setup_client(self):
        """Configure MQTT client callbacks"""
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
    def _on_connect(self, client, userdata, flags, rc):
        """Handle connection established"""
        if rc == 0:
            logger.info("📡 Connected to MQTT broker")
            self.connected = True
            
            # Subscribe to all relevant topics
            self.client.subscribe([
                ("db/+/queries", 1),  # Database queries
                ("db/+/updates", 1),  # Database updates
                ("scraper/+/tasks", 1),  # Scraper task info
                ("scraper/+/results", 1)  # Scraper results
            ])
        else:
            logger.error(f"❌ Failed to connect to MQTT broker: {rc}")
            
    def _on_message(self, client, userdata, msg: MQTTMessage):
        """Handle incoming MQTT messages"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload)
            
            # Find and execute registered handler
            handler = self._get_topic_handler(topic)
            if handler:
                asyncio.create_task(handler(topic, payload))
            else:
                logger.warning(f"⚠️ No handler for topic: {topic}")
                
        except json.JSONDecodeError:
            logger.error("❌ Invalid JSON payload received")
        except Exception as e:
            logger.error(f"❌ Error handling message: {str(e)}")
            
    def _on_disconnect(self, client, userdata, rc):
        """Handle disconnection"""
        self.connected = False
        logger.warning("📡 Disconnected from MQTT broker")
        
    def _get_topic_handler(self, topic: str) -> Optional[Callable]:
        """Get handler for topic pattern"""
        for pattern, handler in self.handlers.items():
            if self._topic_matches(pattern, topic):
                return handler
        return None
        
    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if topic matches pattern with wildcards"""
        pattern_parts = pattern.split('/')
        topic_parts = topic.split('/')
        
        if len(pattern_parts) != len(topic_parts):
            return False
            
        return all(
            p == '+' or p == '#' or p == t
            for p, t in zip(pattern_parts, topic_parts)
        )
        
    def register_handler(self, topic_pattern: str, handler: Callable):
        """
        Register handler for topic pattern
        
        Args:
            topic_pattern: Topic pattern with wildcards
            handler: Async callback(topic, payload)
        """
        self.handlers[topic_pattern] = handler
        
    def start(self):
        """Start MQTT client in background thread"""
        def run_client():
            self.client.connect(self.broker_host, self.broker_port)
            self.client.loop_forever()
            
        self.client_thread = threading.Thread(target=run_client)
        self.client_thread.daemon = True
        self.client_thread.start()
        
    async def publish(self, topic: str, payload: Dict, qos: int = 1):
        """
        Publish message to topic
        
        Args:
            topic: Target topic
            payload: Message payload (will be JSON encoded)
            qos: Quality of Service level
        """
        if not self.connected:
            logger.error("❌ Cannot publish: Not connected")
            return
            
        try:
            message = json.dumps(payload)
            self.client.publish(topic, message, qos)
            logger.debug(f"📤 Published to {topic}")
        except Exception as e:
            logger.error(f"❌ Error publishing message: {str(e)}")
            
    async def close(self):
        """Disconnect client"""
        self.client.disconnect()
        if hasattr(self, 'client_thread'):
            self.client_thread.join(timeout=1.0)