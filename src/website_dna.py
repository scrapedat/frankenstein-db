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
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import msgpack
import lzma

# Framework detection flags
REACT_FLAG = 1 << 0
ANGULAR_FLAG = 1 << 1
VUE_FLAG = 1 << 2
SVELTE_FLAG = 1 << 3
SPA_FLAG = 1 << 4
SSR_FLAG = 1 << 5

logger = logging.getLogger(__name__)


@dataclass
class ScrapingStrategy:
    """Strategy for scraping a specific website with success metrics"""
    name: str
    strategy_type: str = 'normal'  # 'use_cache', 'discovery', 'normal', 'careful'
    stealth_level: str = 'medium'  # 'low', 'medium', 'high', 'maximum'
    success_rate: float = 0.0
    total_attempts: int = 0
    avg_execution_time: float = 0.0
    last_success: Optional[float] = None  # timestamp
    failure_patterns: Dict[str, int] = field(default_factory=dict)
    
    # Strategy configuration
    delay_ms: int = 1500
    timeout_seconds: int = 30
    retry_attempts: int = 2
    use_proxy: bool = False
    known_selectors: Dict = field(default_factory=dict)
    estimated_time_seconds: int = 30
    reason: str = ""
    
    def update_metrics(self, success: bool, execution_time: float, error: Optional[str] = None):
        """Update strategy metrics after an attempt"""
        self.total_attempts += 1
        
        if success:
            self.last_success = time.time()
            # Update success rate with exponential moving average
            alpha = 0.1  # Weight for recent results
            self.success_rate = (1 - alpha) * self.success_rate + alpha
        elif error:
            self.failure_patterns[error] = self.failure_patterns.get(error, 0) + 1
            
        # Update average execution time
        self.avg_execution_time = (
            (self.avg_execution_time * (self.total_attempts - 1) + execution_time) 
            / self.total_attempts
        )
        
        # Adjust configuration based on results
        if not success:
            self.timeout_seconds = min(120, self.timeout_seconds * 1.5)  # Increase timeout
            self.delay_ms = min(10000, int(self.delay_ms * 1.2))  # Increase delay
            
            if 'captcha' in str(error).lower():
                self.stealth_level = 'maximum'
                self.use_proxy = True
            elif 'timeout' in str(error).lower():
                self.timeout_seconds = min(180, self.timeout_seconds * 2)


@dataclass
class WebsiteDNA:
    """
    Compressed website DNA structure with evolution tracking
    
    Represents a website's structure, frameworks, characteristics,
    and plan-execute optimization data in a highly compressed format.
    Includes versioning and mutation prediction capabilities.
    """
    # Core identification
    domain: str
    timestamp: float = field(default_factory=time.time)
    structure_hash: str = ""
    page_type: str = "unknown"
    version: str = "1.0.0"
    
    # Evolution tracking
    parent_hash: Optional[str] = None
    mutation_score: float = 0.0
    generation: int = 1
    mutation_history: List[Dict] = field(default_factory=list)
    
    # Technical characteristics
    framework_flags: int = 0
    element_signature: bytes = field(default_factory=bytes)
    accessibility_score: float = 0.0
    performance_hints: Dict[str, Any] = field(default_factory=dict)
    
    # Structure DNA
    page_structure: Dict = field(default_factory=dict)
    selector_stability: Dict[str, float] = field(default_factory=dict)
    layout_changes_detected: int = 0
    
    # Performance DNA
    avg_load_time_ms: int = 3000
    success_rate: float = 0.0
    error_patterns: Dict[str, int] = field(default_factory=dict)
    performance_trends: List[Dict] = field(default_factory=list)
    
    # Anti-bot DNA
    captcha_frequency: float = 0.0
    rate_limit_threshold: int = 100
    
    # Mutation prediction
    mutation_probability: float = 0.0
    predicted_changes: List[Dict] = field(default_factory=list)
    stability_score: float = 1.0
    change_frequency: Dict[str, float] = field(default_factory=dict)
    
    # Machine learning features
    feature_vectors: Dict[str, List[float]] = field(default_factory=dict)
    pattern_clusters: Dict[str, str] = field(default_factory=dict)
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    stealth_requirements: Dict[str, Any] = field(default_factory=lambda: {'level': 'medium'})
    
    # Content DNA
    content_update_patterns: Dict = field(default_factory=dict)
    data_freshness_hours: int = 24
    last_scraped: float = field(default_factory=time.time)
    scrape_frequency_hours: int = 24
    
    # Plan-Execute optimization
    strategies: Dict[str, ScrapingStrategy] = field(default_factory=dict)
    plan_patterns: List[Dict] = field(default_factory=list)
    framework_specific_steps: Dict[str, List[Dict]] = field(default_factory=dict)

    def compress(self) -> bytes:
        """Compress DNA for storage using MessagePack + LZMA"""
        data = {
            'd': self.domain,
            't': int(self.timestamp),
            'h': self.structure_hash[:16],
            'p': self.page_type[:10],
            'f': self.framework_flags,
            'e': self.element_signature,
            'perf': self.performance_hints,
            'struct': self.page_structure,
            'stab': self.selector_stability,
            'changes': self.layout_changes_detected,
            'load': self.avg_load_time_ms,
            'success': float(self.success_rate),
            'errors': self.error_patterns,
            'captcha': float(self.captcha_frequency),
            'rate_limit': self.rate_limit_threshold,
            'stealth': self.stealth_requirements,
            'updates': self.content_update_patterns,
            'fresh': self.data_freshness_hours,
            'last': int(self.last_scraped),
            'freq': self.scrape_frequency_hours,
            'strat': {
                name: {
                    'type': s.strategy_type,
                    'level': s.stealth_level,
                    'success': float(s.success_rate),
                    'attempts': s.total_attempts,
                    'time': float(s.avg_execution_time),
                    'last': s.last_success,
                    'fails': s.failure_patterns,
                    'delay': s.delay_ms,
                    'timeout': s.timeout_seconds,
                    'retry': s.retry_attempts,
                    'proxy': s.use_proxy,
                    'known': s.known_selectors,
                    'est': s.estimated_time_seconds,
                    'reason': s.reason
                }
                for name, s in self.strategies.items()
            }
        }
        return lzma.compress(msgpack.packb(data))

    @classmethod
    def decompress(cls, data: bytes) -> 'WebsiteDNA':
        """Reconstruct DNA from compressed storage"""
        unpacked = msgpack.unpackb(lzma.decompress(data))
        
        # Create base DNA
        dna = cls(domain=unpacked['d'])
        dna.timestamp = unpacked['t']
        dna.structure_hash = unpacked['h']
        dna.page_type = unpacked['p']
        dna.framework_flags = unpacked['f']
        dna.element_signature = unpacked['e']
        dna.performance_hints = unpacked['perf']
        dna.page_structure = unpacked['struct']
        dna.selector_stability = unpacked['stab']
        dna.layout_changes_detected = unpacked['changes']
        dna.avg_load_time_ms = unpacked['load']
        dna.success_rate = unpacked['success']
        dna.error_patterns = unpacked['errors']
        dna.captcha_frequency = unpacked['captcha']
        dna.rate_limit_threshold = unpacked['rate_limit']
        dna.stealth_requirements = unpacked['stealth']
        dna.content_update_patterns = unpacked['updates']
        dna.data_freshness_hours = unpacked['fresh']
        dna.last_scraped = unpacked['last']
        dna.scrape_frequency_hours = unpacked['freq']
        
        # Reconstruct strategies
        for name, strat in unpacked.get('strat', {}).items():
            dna.strategies[name] = ScrapingStrategy(
                name=name,
                strategy_type=strat['type'],
                stealth_level=strat['level'],
                success_rate=strat['success'],
                total_attempts=strat['attempts'],
                avg_execution_time=strat['time'],
                last_success=strat['last'],
                failure_patterns=strat['fails'],
                delay_ms=strat['delay'],
                timeout_seconds=strat['timeout'],
                retry_attempts=strat['retry'],
                use_proxy=strat['proxy'],
                known_selectors=strat['known'],
                estimated_time_seconds=strat['est'],
                reason=strat['reason']
            )
        
        return dna

    def is_fresh(self) -> bool:
        """Check if the data is still fresh based on update patterns"""
        age_hours = (time.time() - self.last_scraped) / 3600
        return age_hours < self.data_freshness_hours
    
    def get_best_strategy(self) -> Tuple[str, Optional[ScrapingStrategy]]:
        """Get the most successful strategy for this website"""
        if not self.strategies:
            return 'discovery', None
            
        # Find strategy with best success rate and speed
        return max(
            self.strategies.items(), 
            key=lambda x: (x[1].success_rate, -x[1].avg_execution_time)
        )
    
    def generate_strategy(self, context: Dict = None) -> ScrapingStrategy:
        """Generate optimal scraping strategy based on DNA state"""
        if self.is_fresh():
            return ScrapingStrategy(
                name='cache',
                strategy_type='use_cache',
                stealth_level='none',
                delay_ms=0,
                estimated_time_seconds=1,
                reason='Data is fresh'
            )
            
        # Check if we have a successful strategy
        strategy_name, best_strategy = self.get_best_strategy()
        if best_strategy and best_strategy.success_rate > 0.8:
            return best_strategy
            
        # Generate strategy based on DNA
        if self.captcha_frequency > 0.3:
            return ScrapingStrategy(
                name='stealth',
                strategy_type='careful',
                stealth_level='maximum',
                delay_ms=8000,
                retry_attempts=2,
                use_proxy=True,
                estimated_time_seconds=90,
                reason=f'High captcha frequency ({self.captcha_frequency:.1f})'
            )
            
        if self.layout_changes_detected > 3:
            return ScrapingStrategy(
                name='adaptive',
                strategy_type='adaptive',
                stealth_level='high',
                delay_ms=3000,
                known_selectors=self.page_structure,
                estimated_time_seconds=self.avg_load_time_ms // 1000 + 20,
                reason=f'Frequent layout changes ({self.layout_changes_detected})'
            )
            
        if self.success_rate < 0.5:
            return ScrapingStrategy(
                name='careful',
                strategy_type='careful',
                stealth_level='high',
                delay_ms=5000,
                retry_attempts=3,
                use_proxy=True,
                timeout_seconds=60,
                estimated_time_seconds=120,
                reason=f'Low success rate ({self.success_rate:.1f}%)'
            )
            
        # Default strategy with DNA-based tweaks
        return ScrapingStrategy(
            name='normal',
            strategy_type='normal',
            stealth_level=self.stealth_requirements['level'],
            delay_ms=max(1500, self.avg_load_time_ms // 10),
            known_selectors=self.page_structure,
            estimated_time_seconds=self.avg_load_time_ms // 1000 + 10,
            reason='Standard approach based on stable DNA'
        )
    
    def update_from_result(self, result: Dict):
        """Update DNA based on scraping results"""
        self.last_scraped = time.time()
        
        # Update success metrics
        success = result.get('success', False)
        old_weight, new_weight = 0.7, 0.3
        self.success_rate = (old_weight * self.success_rate + new_weight * float(success))
        
        # Update load time
        if 'load_time_ms' in result:
            self.avg_load_time_ms = int(
                old_weight * self.avg_load_time_ms + 
                new_weight * result['load_time_ms']
            )
        
        # Update selectors and track changes
        if 'selectors_used' in result:
            self.update_structure(result['selectors_used'])
        
        # Update anti-bot measures
        errors = result.get('errors', [])
        for error in errors:
            error_str = str(error).lower()
            if 'captcha' in error_str:
                self.captcha_frequency += 0.1
                self.stealth_requirements['level'] = 'high'
            elif 'rate limit' in error_str:
                self.rate_limit_threshold = max(50, self.rate_limit_threshold - 10)
            
            # Track error patterns
            self.error_patterns[error_str] = self.error_patterns.get(error_str, 0) + 1
        
        # Update strategy metrics
        strategy_name = result.get('strategy', 'default')
        if strategy_name in self.strategies:
            self.strategies[strategy_name].update_metrics(
                success=success,
                execution_time=result.get('execution_time', 0.0),
                error=result.get('error')
            )
    
    def update_structure(self, new_structure: Dict):
        """Update page structure and track changes"""
        if self.page_structure:
            # Calculate similarity score
            old_selectors = set(self.page_structure.keys())
            new_selectors = set(new_structure.keys())
            
            changed = len(old_selectors.symmetric_difference(new_selectors))
            total = len(old_selectors.union(new_selectors))
            
            if total > 0:
                change_ratio = changed / total
                if change_ratio > 0.1:  # More than 10% changed
                    self.layout_changes_detected += 1
                    
                    # Track selector evolution
                    for selector_type, old_selector in self.page_structure.items():
                        new_selector = new_structure.get(selector_type)
                        if new_selector and new_selector != old_selector:
                            if 'selector_evolution' not in self.content_update_patterns:
                                self.content_update_patterns['selector_evolution'] = []
                                
                            self.content_update_patterns['selector_evolution'].append({
                                'selector_type': selector_type,
                                'old_selector': old_selector,
                                'new_selector': new_selector,
                                'confidence_score': self.success_rate,
                                'detected_at': time.time()
                            })
                            
                            # Update selector stability score
                            if selector_type in self.selector_stability:
                                self.selector_stability[selector_type] *= 0.8
                            else:
                                self.selector_stability[selector_type] = 0.8
        else:
            # First time seeing this structure
            self.selector_stability = {k: 1.0 for k in new_structure.keys()}
        
        self.page_structure = new_structure
        self.structure_hash = hashlib.sha256(
            json.dumps(new_structure, sort_keys=True).encode()
        ).hexdigest()
    
    def get_analytics(self) -> Dict:
        """Get comprehensive analytics about this website's DNA"""
        return {
            'domain': self.domain,
            'last_update': self.last_scraped,
            'success_metrics': {
                'success_rate': self.success_rate,
                'avg_load_time_ms': self.avg_load_time_ms,
                'total_strategies': len(self.strategies),
                'best_strategy': self.get_best_strategy()[0]
            },
            'structure_health': {
                'layout_changes': self.layout_changes_detected,
                'selector_count': len(self.page_structure),
                'selector_evolution_count': len(
                    self.content_update_patterns.get('selector_evolution', [])
                ),
                'selector_stability': self.selector_stability
            },
            'protection_metrics': {
                'captcha_frequency': self.captcha_frequency,
                'rate_limit_threshold': self.rate_limit_threshold,
                'stealth_level': self.stealth_requirements.get('level', 'medium')
            },
            'freshness': {
                'scrape_frequency_hours': self.scrape_frequency_hours,
                'data_freshness_hours': self.data_freshness_hours,
                'is_fresh': self.is_fresh(),
                'age_hours': (time.time() - self.last_scraped) / 3600
            },
            'technical': {
                'framework_flags': self.framework_flags,
                'page_type': self.page_type,
                'performance_hints': self.performance_hints
            }
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
def predict_mutations(dna: WebsiteDNA, history: List[Dict]) -> Dict:
    """
    Predict likely mutations based on historical patterns
    
    Args:
        dna: Current DNA instance
        history: Historical mutations
        
    Returns:
        Dictionary of predicted changes and probabilities
    """
    predictions = {
        'structure_changes': [],
        'framework_updates': [],
        'probability': 0.0
    }
    
    try:
        # Analyze historical patterns
        changes = _analyze_mutation_history(history)
        
        # Calculate mutation probabilities
        prob = _calculate_mutation_probability(dna, changes)
        predictions['probability'] = prob
        
        if prob > 0.3:  # Significant chance of mutation
            predictions.update(_predict_specific_changes(dna, changes))
            
    except Exception as e:
        logger.error(f"Error predicting mutations: {e}")
        
    return predictions


def _analyze_mutation_history(history: List[Dict]) -> Dict:
    """
    Analyze historical mutation patterns
    
    Args:
        history: List of historical mutations
        
    Returns:
        Analysis of mutation patterns
    """
    patterns = {
        'frequency': {},
        'correlations': {},
        'seasonality': {}
    }
    
    for mutation in history:
        # Track change frequencies
        change_type = mutation.get('type')
        if change_type:
            patterns['frequency'][change_type] = \
                patterns['frequency'].get(change_type, 0) + 1
                
        # Find correlated changes
        changes = mutation.get('changes', [])
        if len(changes) > 1:
            for i, change in enumerate(changes[:-1]):
                key = f"{change['type']}:{changes[i+1]['type']}"
                patterns['correlations'][key] = \
                    patterns['correlations'].get(key, 0) + 1
                    
    return patterns


def _calculate_mutation_probability(dna: WebsiteDNA,
                                changes: Dict) -> float:
    """
    Calculate probability of mutation
    
    Args:
        dna: Current DNA instance
        changes: Analyzed change patterns
        
    Returns:
        Probability of mutation (0-1)
    """
    probability = 0.0
    
    # Base probability from stability score
    probability += (1 - dna.stability_score) * 0.4
    
    # Adjust for change frequency
    avg_changes = sum(dna.change_frequency.values()) / \
        max(len(dna.change_frequency), 1)
    probability += min(avg_changes / 10, 0.3)
    
    # Consider historical patterns
    if changes['frequency']:
        pattern_strength = sum(changes['correlations'].values()) / \
            sum(changes['frequency'].values())
        probability += pattern_strength * 0.3
        
    return min(probability, 1.0)


def _predict_specific_changes(dna: WebsiteDNA, 
                           changes: Dict) -> Dict:
    """
    Predict specific mutation types
    
    Args:
        dna: Current DNA instance
        changes: Analyzed change patterns
        
    Returns:
        Dictionary of predicted changes
    """
    predictions = {
        'structure_changes': [],
        'framework_updates': []
    }
    
    # Predict structure changes
    if 'structure' in changes['frequency']:
        predictions['structure_changes'].extend(
            _predict_structure_mutations(dna, changes)
        )
        
    # Predict framework updates
    if 'framework' in changes['frequency']:
        predictions['framework_updates'].extend(
            _predict_framework_mutations(dna, changes)
        )
        
    return predictions


def generate_proof_hash(url: str, content: str) -> str:
    """Generate proof-of-scraping hash for verification"""
    proof_data = f"{url}:{len(content)}:{int(time.time())}"
    return hashlib.blake2b(proof_data.encode(), digest_size=6).hexdigest()