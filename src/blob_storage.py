"""
Blob Storage - File-based storage for web content

Efficient file-based storage system for cached web content with
automatic compression and nested directory structure for scalability.
"""

import os
import asyncio
import hashlib
import gzip
import logging
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class BlobStorage:
    """
    Simple file-based blob storage (MinIO emulator)
    
    Provides efficient storage for web content with automatic compression,
    nested directory structure, and metadata tracking.
    """
    
    def __init__(self, storage_path: str = "./blob_storage",
                 cache_size_mb: int = 1024,
                 dna_aware: bool = True):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_path / "metadata.json"
        self.cache_size_mb = cache_size_mb
        self.dna_aware = dna_aware
        
        # Content categorization
        self.category_patterns = {
            'html': ['.html', '.htm', 'text/html'],
            'json': ['.json', 'application/json'],
            'image': ['image/', '.jpg', '.png', '.gif'],
            'script': ['.js', '.jsx', '.ts', '.tsx'],
            'style': ['.css', '.scss', '.sass']
        }
        
        # Cache structures
        self.content_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # DNA patterns for optimization
        self.dna_patterns = {}
        self.access_patterns = {}
        
        self._init_metadata()
        logger.info(f"💾 Blob storage initialized at {self.storage_path}")

    def _init_metadata(self):
        """Initialize metadata tracking"""
        if not self.metadata_file.exists():
            import json
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    'total_blobs': 0,
                    'total_size': 0,
                    'created': str(self.storage_path),
                    'last_cleanup': None
                }, f)

    def _categorize_content(self, key: str, data: bytes) -> str:
        """
        Categorize content for optimal compression
        
        Args:
            key: Content identifier
            data: Raw content data
            
        Returns:
            Content category
        """
        # Check file extension and content type
        for category, patterns in self.category_patterns.items():
            if any(p in key.lower() for p in patterns):
                return category
                
        # Analyze content heuristically
        try:
            sample = data[:1024].decode('utf-8', errors='ignore')
            
            if '<html' in sample or '<!DOCTYPE' in sample:
                return 'html'
            elif '{' in sample and ':' in sample:
                return 'json'
            elif 'function' in sample or 'var' in sample:
                return 'script'
            elif '{' in sample and ';' in sample:
                return 'style'
        except:
            pass
            
        return 'binary'
    
    def _select_compression(self, category: str, size: int) -> Tuple[str, Dict]:
        """
        Select optimal compression method
        
        Args:
            category: Content category
            size: Content size in bytes
            
        Returns:
            Tuple of (compression_method, parameters)
        """
        compression_map = {
            'html': ('zstd', {'level': 3}),
            'json': ('zstd', {'level': 4}),
            'script': ('zstd', {'level': 5}),
            'style': ('zstd', {'level': 4}),
            'image': ('lz4', {'acceleration': 2}),
            'binary': ('lz4', {'acceleration': 1})
        }
        
        # Get base configuration
        method, params = compression_map.get(category, ('zstd', {'level': 1}))
        
        # Adjust based on size
        if size > 1024 * 1024:  # > 1MB
            if method == 'zstd':
                params['level'] = min(params['level'] + 1, 9)
            else:
                params['acceleration'] = max(params['acceleration'] - 1, 1)
                
        return method, params
    
    async def store_blob(self, key: str, data: bytes, compress: bool = True, metadata: Optional[Dict] = None) -> str:
        """
        Store blob data with intelligent compression
        
        Args:
            key: Unique identifier for the blob
            data: Raw data to store
            compress: Whether to compress the data (default True)
            metadata: Optional metadata to store alongside
            
        Returns:
            File path where blob was stored
        """
        # Compress data if requested
        if compress:
            data = gzip.compress(data)
        
        # Create nested directory structure for scalability
        hash_key = hashlib.md5(key.encode()).hexdigest()
        dir_path = self.storage_path / hash_key[:2] / hash_key[2:4]
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Store main blob
        file_path = dir_path / f"{hash_key}.blob"
        
        def _write():
            with open(file_path, 'wb') as f:
                f.write(data)
            
            # Store metadata if provided
            if metadata:
                metadata_path = dir_path / f"{hash_key}.meta"
                import json
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'key': key,
                        'compressed': compress,
                        'size': len(data),
                        'original_size': metadata.get('original_size', len(data)),
                        'content_type': metadata.get('content_type'),
                        'timestamp': metadata.get('timestamp'),
                        'url': metadata.get('url')
                    }, f)
        
        await asyncio.get_event_loop().run_in_executor(None, _write)
        return str(file_path)

    def _update_access_pattern(self, key: str):
        """
        Update content access patterns
        
        Args:
            key: Accessed content key
        """
        now = time.time()
        
        if key not in self.access_patterns:
            self.access_patterns[key] = {
                'count': 0,
                'last_access': now,
                'intervals': [],
                'associated_keys': set()
            }
            
        pattern = self.access_patterns[key]
        
        # Update statistics
        pattern['count'] += 1
        if pattern['last_access']:
            interval = now - pattern['last_access']
            pattern['intervals'].append(interval)
            pattern['intervals'] = pattern['intervals'][-10:]  # Keep last 10
            
        pattern['last_access'] = now
        
        # Cleanup old patterns
        self._cleanup_access_patterns()
    
    def _predict_next_access(self, key: str) -> List[str]:
        """
        Predict next likely content accesses
        
        Args:
            key: Current content key
            
        Returns:
            List of content keys likely to be accessed next
        """
        if key not in self.access_patterns:
            return []
            
        pattern = self.access_patterns[key]
        predictions = []
        
        # Check associated content
        for associated in pattern['associated_keys']:
            if associated in self.access_patterns:
                predictions.append((associated, 
                                  self.access_patterns[associated]['count']))
                
        # Sort by access count
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return [p[0] for p in predictions[:5]]
    
    async def get_blob(self, key: str, decompress: bool = True) -> Optional[bytes]:
        """
        Retrieve blob data with predictive caching
        
        Args:
            key: Blob identifier
            decompress: Whether to decompress the data (default True)
            
        Returns:
            Blob data or None if not found
        """
        hash_key = hashlib.md5(key.encode()).hexdigest()
        file_path = self.storage_path / hash_key[:2] / hash_key[2:4] / f"{hash_key}.blob"
        
        def _read():
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                    return gzip.decompress(data) if decompress else data
            except FileNotFoundError:
                return None
            except gzip.BadGzipFile:
                # Data wasn't compressed, return as-is
                with open(file_path, 'rb') as f:
                    return f.read()
        
        return await asyncio.get_event_loop().run_in_executor(None, _read)

    async def get_blob_metadata(self, key: str) -> Optional[Dict]:
        """
        Get metadata for a blob
        
        Args:
            key: Blob identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        hash_key = hashlib.md5(key.encode()).hexdigest()
        metadata_path = self.storage_path / hash_key[:2] / hash_key[2:4] / f"{hash_key}.meta"
        
        def _read():
            try:
                import json
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                return None
        
        return await asyncio.get_event_loop().run_in_executor(None, _read)

    async def blob_exists(self, key: str) -> bool:
        """
        Check if a blob exists
        
        Args:
            key: Blob identifier
            
        Returns:
            True if blob exists, False otherwise
        """
        hash_key = hashlib.md5(key.encode()).hexdigest()
        file_path = self.storage_path / hash_key[:2] / hash_key[2:4] / f"{hash_key}.blob"
        return file_path.exists()

    async def delete_blob(self, key: str) -> bool:
        """
        Delete a blob and its metadata
        
        Args:
            key: Blob identifier
            
        Returns:
            True if blob was deleted, False if it didn't exist
        """
        hash_key = hashlib.md5(key.encode()).hexdigest()
        dir_path = self.storage_path / hash_key[:2] / hash_key[2:4]
        blob_path = dir_path / f"{hash_key}.blob"
        meta_path = dir_path / f"{hash_key}.meta"
        
        def _delete():
            deleted = False
            try:
                if blob_path.exists():
                    blob_path.unlink()
                    deleted = True
                if meta_path.exists():
                    meta_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete blob {key}: {e}")
            return deleted
        
        return await asyncio.get_event_loop().run_in_executor(None, _delete)

    async def list_blobs(self, prefix: str = "", limit: int = 100) -> List[str]:
        """
        List stored blobs with optional prefix filter
        
        Args:
            prefix: Key prefix to filter by
            limit: Maximum number of results
            
        Returns:
            List of blob keys
        """
        def _list():
            keys = []
            count = 0
            
            for blob_file in self.storage_path.rglob("*.blob"):
                if count >= limit:
                    break
                
                # Try to get the original key from metadata
                hash_key = blob_file.stem
                meta_file = blob_file.parent / f"{hash_key}.meta"
                
                if meta_file.exists():
                    try:
                        import json
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                            key = metadata.get('key', hash_key)
                            if key.startswith(prefix):
                                keys.append(key)
                                count += 1
                    except:
                        # If metadata is corrupted, use hash as key
                        if hash_key.startswith(prefix):
                            keys.append(hash_key)
                            count += 1
                else:
                    # No metadata, use hash as key
                    if hash_key.startswith(prefix):
                        keys.append(hash_key)
                        count += 1
            
            return keys
        
        return await asyncio.get_event_loop().run_in_executor(None, _list)

    async def get_storage_stats(self) -> Dict:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        def _stats():
            total_files = 0
            total_size = 0
            blob_files = 0
            meta_files = 0
            
            for file_path in self.storage_path.rglob("*"):
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size
                    
                    if file_path.suffix == '.blob':
                        blob_files += 1
                    elif file_path.suffix == '.meta':
                        meta_files += 1
            
            return {
                'total_files': total_files,
                'blob_files': blob_files,
                'metadata_files': meta_files,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'storage_path': str(self.storage_path)
            }
        
        return await asyncio.get_event_loop().run_in_executor(None, _stats)

    async def cleanup_orphaned_files(self) -> Dict:
        """
        Clean up orphaned metadata files and empty directories
        
        Returns:
            Dictionary with cleanup statistics
        """
        def _cleanup():
            orphaned_meta = 0
            empty_dirs = 0
            
            # Find orphaned metadata files (meta without corresponding blob)
            for meta_file in self.storage_path.rglob("*.meta"):
                blob_file = meta_file.parent / f"{meta_file.stem}.blob"
                if not blob_file.exists():
                    try:
                        meta_file.unlink()
                        orphaned_meta += 1
                    except OSError:
                        pass
            
            # Remove empty directories (bottom-up)
            for dir_path in sorted(self.storage_path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if dir_path.is_dir() and dir_path != self.storage_path:
                    try:
                        dir_path.rmdir()  # Only succeeds if directory is empty
                        empty_dirs += 1
                    except OSError:
                        pass  # Directory not empty or other error
            
            return {
                'orphaned_metadata_removed': orphaned_meta,
                'empty_directories_removed': empty_dirs
            }
        
        return await asyncio.get_event_loop().run_in_executor(None, _cleanup)

    async def store_website_content(self, url: str, html_content: str, content_type: str = "text/html") -> str:
        """
        Convenience method to store website content with proper metadata
        
        Args:
            url: Website URL
            html_content: HTML content
            content_type: MIME type of content
            
        Returns:
            Storage path
        """
        import time
        
        key = f"html:{url}"
        metadata = {
            'url': url,
            'content_type': content_type,
            'timestamp': time.time(),
            'original_size': len(html_content.encode('utf-8'))
        }
        
        return await self.store_blob(
            key, 
            html_content.encode('utf-8'), 
            compress=True, 
            metadata=metadata
        )

    async def get_website_content(self, url: str) -> Optional[str]:
        """
        Convenience method to retrieve website content
        
        Args:
            url: Website URL
            
        Returns:
            HTML content as string or None if not found
        """
        key = f"html:{url}"
        data = await self.get_blob(key, decompress=True)
        return data.decode('utf-8') if data else None