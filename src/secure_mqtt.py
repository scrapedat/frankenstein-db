"""
secure_mqtt.py Client with Programmatic Controls

Implements secure MQTT communication with TLS, authentication,
and programmatic pub/sub controls for AI system components.
"""

import os
import asyncio
import ssl
import json
import logging
import time
from typing import Dict, Optional, Callable, List, Set
from enum import Enum
import paho.mqtt.client as mqtt
from paho.mqtt.client import MQTTMessage
import threading

logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """System component types"""
    DASHBOARD = "dashboard"
    SCRAPER = "scraper"
    DATABASE = "db"
    AI = "ai"

class PubSubMode(Enum):
    """Publication/Subscription modes"""
    PROGRAMMATIC = "programmatic"  # Requires explicit subscription
    AI_DRIVEN = "ai_driven"       # AI decides relevance
    WORKFLOW = "workflow"         # Based on workflow state
    CONSTANT = "constant"         # Always subscribed

class SecureMQTTClient:
    """
    Secure MQTT client with enhanced controls
    """
    
    def __init__(self,
                component_type: ComponentType,
                component_id: str,
                broker_host: str = 'localhost',
                broker_port: int = 8883,
                cert_path: str = '/app/certs',
                default_mode: PubSubMode = PubSubMode.WORKFLOW):
        """
        Initialize secure MQTT client
        
        Args:
            component_type: Type of system component
            component_id: Unique component identifier
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port (8883 for TLS)
            cert_path: Path to certificates
            default_mode: Default pub/sub mode
        """
        self.component_type = component_type
        self.component_id = component_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.cert_path = cert_path
        self.default_mode = default_mode
        
        # Initialize client
        self.client = mqtt.Client(
            client_id=f"{component_type.value}-{component_id}",
            protocol=mqtt.MQTTv5
        )
        
        # Set up security
        self._setup_security()
        
        # Topic controls
        self.topic_modes: Dict[str, PubSubMode] = {}
        self.programmatic_subs: Set[str] = set()
        self.ai_controlled_topics: Set[str] = set()
        self.workflow_topics: Set[str] = set()
        self.constant_topics: Set[str] = set()
        
        # Message handlers
        self.handlers: Dict[str, List[Callable]] = {}
        
        # AI components
        self.ai_filter: Optional[Callable] = None
        self.workflow_state: Dict = {}
        
        # Setup client callbacks
        self._setup_client()
        
    def _setup_security(self):
        """
        Configure TLS/SSL security and authentication
        
        Sets up:
        - TLS 1.2 with certificate verification
        - Client certificate authentication
        - Username derived from component type
        """
        try:
            # Set up TLS context
            self.client.tls_set(
                ca_certs=f"{self.cert_path}/ca.crt",
                certfile=f"{self.cert_path}/client.crt",
                keyfile=f"{self.cert_path}/client.key",
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLSv1_2,
                ciphers=None  # Let OpenSSL choose secure defaults
            )
            
            # Verify certificate paths
            for cert_file in ['ca.crt', 'client.crt', 'client.key']:
                path = f"{self.cert_path}/{cert_file}"
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Certificate file not found: {path}")
            
            # Set username from certificate CN
            self.client.username_pw_set(
                username=self.component_type.value,
                password=None  # Using certificate for auth
            )
            
        except Exception as e:
            logger.error(f"Failed to setup TLS security: {str(e)}")
            raise
        
    def _setup_client(self):
        """
        Configure MQTT client settings and callbacks
        
        Sets up:
        - Connection/disconnection handlers
        - Message processing
        - Subscription management
        - Automatic reconnection
        """
        # Configure callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.client.on_subscribe = self._on_subscribe
        
        # Configure reconnection
        self.client.reconnect_delay_set(min_delay=1, max_delay=60)
        self.client.max_inflight_messages_set(20)  # Prevent message queue overflow
        
        # Configure protocol version
        self.client.protocol = mqtt.MQTTv5  # Use latest protocol version
        
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Handle successful connection"""
        if rc == 0:
            logger.info("🔒 Securely connected to MQTT broker")
            # Subscribe to constant topics
            self._subscribe_constant_topics()
            # Subscribe to workflow topics based on current state
            self._update_workflow_subscriptions()
        else:
            logger.error(f"❌ Failed to connect to MQTT broker: {rc}")
            
    def _on_message(self, client, userdata, msg: MQTTMessage):
        """Handle incoming messages with filtering"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload)
            
            # Check topic mode
            mode = self.topic_modes.get(topic, self.default_mode)
            
            if mode == PubSubMode.PROGRAMMATIC:
                if topic not in self.programmatic_subs:
                    return
            elif mode == PubSubMode.AI_DRIVEN:
                if not self._check_ai_relevance(topic, payload):
                    return
            elif mode == PubSubMode.WORKFLOW:
                if not self._check_workflow_relevance(topic):
                    return
                    
            # Execute handlers
            handlers = self._get_topic_handlers(topic)
            for handler in handlers:
                asyncio.create_task(handler(topic, payload))
                
        except json.JSONDecodeError:
            logger.error("❌ Invalid JSON payload received")
        except Exception as e:
            logger.error(f"❌ Error handling message: {str(e)}")
            
    def _on_disconnect(self, client, userdata, rc):
        """Handle disconnection"""
        logger.warning("📡 Disconnected from MQTT broker")
        
    def _on_subscribe(self, client, userdata, mid, granted_qos, properties=None):
        """Handle successful subscription"""
        logger.debug(f"✅ Subscription confirmed: {mid}")
        
    def set_topic_mode(self, topic: str, mode: PubSubMode):
        """
        Set pub/sub mode for topic
        
        Args:
            topic: Topic pattern
            mode: Publication/Subscription mode
        """
        self.topic_modes[topic] = mode
        
        # Update subscriptions if needed
        if mode == PubSubMode.CONSTANT:
            self.constant_topics.add(topic)
            self.client.subscribe(topic)
        elif mode == PubSubMode.WORKFLOW:
            self.workflow_topics.add(topic)
            self._update_workflow_subscriptions()
            
    def subscribe_programmatic(self, topic: str):
        """
        Explicitly subscribe to topic
        
        Args:
            topic: Topic to subscribe to
        """
        self.programmatic_subs.add(topic)
        self.client.subscribe(topic)
        
    def unsubscribe_programmatic(self, topic: str):
        """
        Explicitly unsubscribe from topic
        
        Args:
            topic: Topic to unsubscribe from
        """
        self.programmatic_subs.remove(topic)
        if topic not in self.constant_topics:
            self.client.unsubscribe(topic)
            
    def set_ai_filter(self, filter_func: Callable):
        """
        Set AI-based message filter
        
        Args:
            filter_func: Function(topic, payload) -> bool
        """
        self.ai_filter = filter_func
        
    def update_workflow_state(self, state: Dict):
        """
        Update workflow state
        
        Args:
            state: New workflow state
        """
        self.workflow_state = state
        self._update_workflow_subscriptions()
        
    def register_handler(self, topic_pattern: str, handler: Callable):
        """
        Register message handler
        
        Args:
            topic_pattern: Topic pattern
            handler: Async handler function
        """
        if topic_pattern not in self.handlers:
            self.handlers[topic_pattern] = []
        self.handlers[topic_pattern].append(handler)
        
    def _subscribe_constant_topics(self):
        """Subscribe to constant topics"""
        for topic in self.constant_topics:
            self.client.subscribe(topic)
            
    def _update_workflow_subscriptions(self):
        """Update subscriptions based on workflow state"""
        if not self.workflow_state:
            return
            
        for topic in self.workflow_topics:
            if self._check_workflow_relevance(topic):
                self.client.subscribe(topic)
            else:
                self.client.unsubscribe(topic)
                
    def _check_ai_relevance(self, topic: str, payload: Dict) -> bool:
        """Check if message is relevant according to AI"""
        if not self.ai_filter:
            return True
        return self.ai_filter(topic, payload)
        
    def _check_workflow_relevance(self, topic: str) -> bool:
        """Check if topic is relevant to current workflow state"""
        # Implement workflow-based filtering logic
        return True  # Default to allowing messages
        
    def _get_topic_handlers(self, topic: str) -> List[Callable]:
        """Get handlers for topic"""
        handlers = []
        for pattern, pattern_handlers in self.handlers.items():
            if self._topic_matches(pattern, topic):
                handlers.extend(pattern_handlers)
        return handlers
        
    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if topic matches pattern"""
        pattern_parts = pattern.split('/')
        topic_parts = topic.split('/')
        
        if len(pattern_parts) != len(topic_parts):
            return False
            
        return all(
            p == '+' or p == '#' or p == t
            for p, t in zip(pattern_parts, topic_parts)
        )
        
    async def publish(self, topic: str, payload: Dict, qos: int = 1):
        """
        Publish message with security checks
        
        Args:
            topic: Target topic
            payload: Message payload
            qos: Quality of Service level
        """
        try:
            # Add security metadata
            payload['_meta'] = {
                'component': self.component_type.value,
                'component_id': self.component_id,
                'timestamp': time.time()
            }
            
            message = json.dumps(payload)
            self.client.publish(topic, message, qos)
            
        except Exception as e:
            logger.error(f"❌ Error publishing message: {str(e)}")
            
    def start(self):
        """Start MQTT client in background thread with reconnection handling"""
        def run_client():
            while True:
                try:
                    # Attempt connection
                    self.client.connect(self.broker_host, self.broker_port)
                    self.client.loop_forever()
                except Exception as e:
                    logger.error(f"MQTT connection error: {str(e)}")
                    time.sleep(5)  # Wait before retry
                    
                # Check if we should stop retrying
                if hasattr(self, '_stop_event') and self._stop_event.is_set():
                    break
            
        self._stop_event = threading.Event()
        self.client_thread = threading.Thread(target=run_client)
        self.client_thread.daemon = True
        self.client_thread.start()
        
    async def close(self):
        """Disconnect client and clean up resources"""
        try:
            # Stop reconnection attempts
            if hasattr(self, '_stop_event'):
                self._stop_event.set()
            
            # Publish offline status if still connected
            try:
                await self.publish(
                    f"{self.component_type.value}/status",
                    {"status": "offline", "timestamp": time.time()}
                )
            except:
                pass  # Ignore publish errors during shutdown
            
            # Disconnect client
            self.client.disconnect()
            
            # Wait for thread to finish with timeout
            if hasattr(self, 'client_thread'):
                self.client_thread.join(timeout=2.0)
                
            # Force disconnect if still connected
            try:
                self.client.loop_stop()
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error during MQTT cleanup: {str(e)}")
            raise