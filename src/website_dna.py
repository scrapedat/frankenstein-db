"""
WebsiteDNA - Compressed website fingerprint structure

This module contains the core data structure for representing website characteristics
in a highly compressed format suitable for time-series storage and analysis.
"""

import json
import time
import hashlib
import struct
import logging
from typing import Dict, Any
from dataclasses import dataclass
import msgpack
import lzma

logger = logging.getLogger(__name__)


@dataclass
class WebsiteDNA:
    """
    Compressed website DNA structure
    
    Represents a website's structure, frameworks, and characteristics
    in a highly compressed format (~1KB) for efficient storage.
    """
    domain: str
    timestamp: float
    structure_hash: str
    page_type: str
    framework_flags: int
    element_signature: bytes
    accessibility_score: float
    performance_hints: Dict[str, Any]
    proof_hash: str

    def compress(self) -> bytes:
        """
        Ultra-compress DNA to ~1KB using MessagePack + LZMA
        
        Returns:
            Compressed bytes representing the DNA
        """
        data = {
            'd': self.domain,
            't': int(self.timestamp),
            'h': self.structure_hash[:16],  # Truncate hash for space
            'p': self.page_type[:10],       # Limit page type length
            'f': self.framework_flags,
            'e': self.element_signature,
            'a': round(self.accessibility_score, 2),
            'perf': {k: round(v, 2) if isinstance(v, float) else v 
                    for k, v in list(self.performance_hints.items())[:5]},  # Top 5 hints
            'proof': self.proof_hash[:12]   # Truncate proof hash
        }
        
        # MessagePack + LZMA compression for optimal size
        packed = msgpack.packb(data, use_bin_type=True)
        return lzma.compress(packed, preset=6)  # Balance speed vs compression

    @classmethod
    def decompress(cls, compressed_data: bytes) -> 'WebsiteDNA':
        """
        Decompress DNA from compressed bytes
        
        Args:
            compressed_data: LZMA compressed DNA data
            
        Returns:
            WebsiteDNA instance
            
        Raises:
            Exception: If decompression fails
        """
        try:
            unpacked = lzma.decompress(compressed_data)
            data = msgpack.unpackb(unpacked, raw=False)
            
            return cls(
                domain=data['d'],
                timestamp=float(data['t']),
                structure_hash=data['h'],
                page_type=data['p'],
                framework_flags=data['f'],
                element_signature=data['e'],
                accessibility_score=data['a'],
                performance_hints=data['perf'],
                proof_hash=data['proof']
            )
        except Exception as e:
            logger.error(f"Failed to decompress DNA: {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert DNA to dictionary for JSON serialization"""
        return {
            'domain': self.domain,
            'timestamp': self.timestamp,
            'structure_hash': self.structure_hash,
            'page_type': self.page_type,
            'framework_flags': self.framework_flags,
            'element_signature': self.element_signature.hex(),  # Convert bytes to hex
            'accessibility_score': self.accessibility_score,
            'performance_hints': self.performance_hints,
            'proof_hash': self.proof_hash,
            'compressed_size': len(self.compress())
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebsiteDNA':
        """Create DNA from dictionary"""
        return cls(
            domain=data['domain'],
            timestamp=data['timestamp'],
            structure_hash=data['structure_hash'],
            page_type=data['page_type'],
            framework_flags=data['framework_flags'],
            element_signature=bytes.fromhex(data['element_signature']),
            accessibility_score=data['accessibility_score'],
            performance_hints=data['performance_hints'],
            proof_hash=data['proof_hash']
        )


def generate_structure_hash(fingerprint: Dict) -> str:
    """Generate compact structure hash from website fingerprint"""
    structure_data = json.dumps({
        'depth': fingerprint.get('dom_depth', 0),
        'elements': fingerprint.get('element_counts', {}),
        'frameworks': sorted(fingerprint.get('framework_signatures', []))
    }, sort_keys=True)
    
    return hashlib.blake2b(structure_data.encode(), digest_size=8).hexdigest()


def encode_frameworks(frameworks: list[str]) -> int:
    """Encode frameworks as bit flags for efficient storage"""
    framework_map = {
        'react': 1, 'vue': 2, 'angular': 4, 'jquery': 8,
        'bootstrap': 16, 'tailwind': 32, 'webpack': 64,
        'gatsby': 128, 'nextjs': 256, 'svelte': 512
    }
    
    flags = 0
    for fw in frameworks:
        if fw in framework_map:
            flags |= framework_map[fw]
    
    return flags


def decode_frameworks(flags: int) -> list[str]:
    """Decode framework bit flags back to list of framework names"""
    framework_map = {
        1: 'react', 2: 'vue', 4: 'angular', 8: 'jquery',
        16: 'bootstrap', 32: 'tailwind', 64: 'webpack',
        128: 'gatsby', 256: 'nextjs', 512: 'svelte'
    }
    
    frameworks = []
    for flag, name in framework_map.items():
        if flags & flag:
            frameworks.append(name)
    
    return frameworks


def compress_elements(element_counts: Dict[str, int]) -> bytes:
    """Compress element counts to binary signature"""
    # Take top 10 most common elements
    sorted_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    signature = b''
    for element, count in sorted_elements:
        # Store as element_name_hash + count (2 bytes each)
        element_hash = hashlib.blake2b(element.encode(), digest_size=1).digest()
        signature += element_hash + struct.pack('B', min(count, 255))
    
    return signature


def calculate_accessibility_score(fingerprint: Dict) -> float:
    """Calculate accessibility score from website fingerprint"""
    features = fingerprint.get('accessibility_features', [])
    total_features = len(features)
    max_score = 5.0  # alt_text, aria, roles, labels, tab_navigation
    
    return min(total_features / max_score, 1.0) * 10.0  # Score out of 10


def generate_proof_hash(url: str, content: str) -> str:
    """Generate proof-of-scraping hash for verification"""
    proof_data = f"{url}:{len(content)}:{int(time.time())}"
    return hashlib.blake2b(proof_data.encode(), digest_size=6).hexdigest()