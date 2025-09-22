"""
FrankensteinDB Production Configuration

Production-ready configuration class for FrankensteinDB with
environment-based settings and production optimizations.
"""

import os
import logging
from typing import Optional
from pathlib import Path

class FrankensteinConfig:
    """Production configuration for FrankensteinDB"""
    
    def __init__(self):
        # Database paths
        self.evolution_db_path = os.getenv(
            'FRANKENSTEIN_EVOLUTION_DB', 
            '/data/sqlite/website_evolution.db'
        )
        self.search_db_path = os.getenv(
            'FRANKENSTEIN_SEARCH_DB', 
            '/data/sqlite/search_index.db'
        )
        self.blob_storage_path = os.getenv(
            'FRANKENSTEIN_BLOB_STORAGE', 
            '/data/blobs'
        )
        
        # Redis configuration
        self.redis_host = os.getenv('FRANKENSTEIN_REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('FRANKENSTEIN_REDIS_PORT', '6379'))
        self.redis_db = int(os.getenv('FRANKENSTEIN_REDIS_DB', '0'))
        
        # Logging configuration
        self.log_level = os.getenv('FRANKENSTEIN_LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('FRANKENSTEIN_LOG_FILE', '/data/logs/frankenstein.log')
        
        # Performance settings
        self.max_blob_size_mb = int(os.getenv('FRANKENSTEIN_MAX_BLOB_SIZE_MB', '100'))
        self.compression_enabled = os.getenv('FRANKENSTEIN_COMPRESSION', 'true').lower() == 'true'
        self.cache_ttl_seconds = int(os.getenv('FRANKENSTEIN_CACHE_TTL', '3600'))
        
        # Security settings
        self.enable_proof_verification = os.getenv('FRANKENSTEIN_PROOF_VERIFICATION', 'true').lower() == 'true'
        self.max_concurrent_operations = int(os.getenv('FRANKENSTEIN_MAX_CONCURRENT', '10'))
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Setup logging
        self._setup_logging()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            Path(self.evolution_db_path).parent,
            Path(self.search_db_path).parent,
            Path(self.blob_storage_path),
            Path(self.log_file).parent if self.log_file else None
        ]
        
        for directory in directories:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup production logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format=log_format,
            handlers=[
                logging.StreamHandler(),  # Console output
                logging.FileHandler(self.log_file) if self.log_file else logging.NullHandler()
            ]
        )
        
        # Set specific logger levels
        logging.getLogger('frankenstein-db').setLevel(getattr(logging, self.log_level.upper()))
        logging.getLogger('redis').setLevel(logging.WARNING)  # Reduce Redis noise
    
    def get_frankenstein_params(self) -> dict:
        """Get parameters for FrankensteinDB initialization"""
        return {
            'evolution_db_path': self.evolution_db_path,
            'search_db_path': self.search_db_path,
            'blob_storage_path': self.blob_storage_path,
            'redis_host': self.redis_host,
            'redis_port': self.redis_port
        }
    
    def __repr__(self):
        return f"FrankensteinConfig(redis={self.redis_host}:{self.redis_port}, db_path={self.evolution_db_path})"


# Production instance
production_config = FrankensteinConfig()


def get_production_frankenstein():
    """Get a production-configured FrankensteinDB instance"""
    from .frankenstein_db import FrankensteinDB
    
    return FrankensteinDB(**production_config.get_frankenstein_params())