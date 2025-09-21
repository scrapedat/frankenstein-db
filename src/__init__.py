"""
FrankensteinDB 🧟‍♂️ - A Beautiful Monster Database System
Stitching together the best parts of different databases for ultimate performance
"""

from .website_dna import WebsiteDNA
from .scylla_emulator import ScyllaDBEmulator
from .redis_store import RedisContextStore
from .search_index import SearchIndex
from .blob_storage import BlobStorage
from .frankenstein_db import FrankensteinDB

__version__ = "0.1.0"
__all__ = [
    "WebsiteDNA",
    "ScyllaDBEmulator", 
    "RedisContextStore",
    "SearchIndex",
    "BlobStorage",
    "FrankensteinDB"
]