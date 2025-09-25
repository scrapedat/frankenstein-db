"""
MQTT Topic Configuration

Defines standard topics and access patterns for system components.
"""

from enum import Enum
from typing import Dict, List

class TopicAccess(Enum):
    """Topic access patterns"""
    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"

class PubSubPatterns:
    """Standard pub/sub patterns for system components"""
    
    # Dashboard patterns
    DASHBOARD_PATTERNS = {
        'commands': {
            'pattern': 'dashboard/{user_id}/commands/#',
            'access': TopicAccess.WRITE,
            'qos': 1
        },
        'status': {
            'pattern': 'dashboard/{user_id}/status',
            'access': TopicAccess.READWRITE,
            'qos': 1
        },
        'responses': {
            'pattern': 'dashboard/{user_id}/responses/#',
            'access': TopicAccess.READ,
            'qos': 1
        }
    }
    
    # Scraper patterns
    SCRAPER_PATTERNS = {
        'tasks': {
            'pattern': 'scraper/{scraper_id}/tasks/#',
            'access': TopicAccess.READ,
            'qos': 1
        },
        'results': {
            'pattern': 'scraper/{scraper_id}/results',
            'access': TopicAccess.WRITE,
            'qos': 1
        },
        'status': {
            'pattern': 'scraper/{scraper_id}/status',
            'access': TopicAccess.WRITE,
            'qos': 1
        },
        'context': {
            'pattern': 'scraper/{scraper_id}/context',
            'access': TopicAccess.WRITE,
            'qos': 1
        }
    }
    
    # Database patterns
    DATABASE_PATTERNS = {
        'queries': {
            'pattern': 'db/{db_id}/queries/#',
            'access': TopicAccess.READ,
            'qos': 1
        },
        'responses': {
            'pattern': 'db/{db_id}/responses/#',
            'access': TopicAccess.WRITE,
            'qos': 1
        },
        'status': {
            'pattern': 'db/{db_id}/status',
            'access': TopicAccess.WRITE,
            'qos': 1
        },
        'insights': {
            'pattern': 'db/insights/#',
            'access': TopicAccess.WRITE,
            'qos': 1
        },
        'health': {
            'pattern': 'db/{db_id}/health',
            'access': TopicAccess.WRITE,
            'qos': 1
        },
        'metrics': {
            'pattern': 'db/{db_id}/metrics/#',
            'access': TopicAccess.WRITE,
            'qos': 1
        },
        'errors': {
            'pattern': 'db/{db_id}/errors',
            'access': TopicAccess.WRITE,
            'qos': 1
        }
    }
    
    # AI workflow patterns
    AI_WORKFLOW_PATTERNS = {
        'tasks': {
            'pattern': 'db/ai/tasks/#',
            'access': TopicAccess.READWRITE,
            'qos': 1
        },
        'insights': {
            'pattern': 'db/ai/insights/#',
            'access': TopicAccess.WRITE,
            'qos': 1
        },
        'context': {
            'pattern': 'db/ai/context/#',
            'access': TopicAccess.WRITE,
            'qos': 1
        },
        'models': {
            'pattern': 'db/ai/models/#',
            'access': TopicAccess.READWRITE,
            'qos': 1
        },
        'training': {
            'pattern': 'db/ai/training/#',
            'access': TopicAccess.WRITE,
            'qos': 1
        },
        'predictions': {
            'pattern': 'db/ai/predictions/#',
            'access': TopicAccess.WRITE,
            'qos': 1
        },
        'monitoring': {
            'pattern': 'db/ai/monitoring/#',
            'access': TopicAccess.WRITE,
            'qos': 1
        },
        'feedback': {
            'pattern': 'db/ai/feedback/#',
            'access': TopicAccess.READWRITE,
            'qos': 1
        }
    }
    
    # Context update patterns
    CONTEXT_PATTERNS = {
        'updates': {
            'pattern': 'db/context/{target_id}/updates',
            'access': TopicAccess.READWRITE,
            'qos': 1
        },
        'feedback': {
            'pattern': 'db/context/{target_id}/feedback',
            'access': TopicAccess.WRITE,
            'qos': 1
        }
    }
    
    @staticmethod
    def get_component_topics(component_type: str, component_id: str) -> Dict[str, Dict]:
        """
        Get topic patterns for component
        
        Args:
            component_type: Type of component
            component_id: Component identifier
            
        Returns:
            Dictionary of topic configurations
        """
        if component_type == 'dashboard':
            return {k: v.copy() for k, v in PubSubPatterns.DASHBOARD_PATTERNS.items()}
        elif component_type == 'scraper':
            return {k: v.copy() for k, v in PubSubPatterns.SCRAPER_PATTERNS.items()}
        elif component_type == 'db':
            patterns = {k: v.copy() for k, v in PubSubPatterns.DATABASE_PATTERNS.items()}
            patterns.update({k: v.copy() for k, v in PubSubPatterns.AI_WORKFLOW_PATTERNS.items()})
            return patterns
        else:
            return {}