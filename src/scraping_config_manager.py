"""
Scraping Configuration Manager

Manages loading, versioning, and synchronization of scraping configurations
from YAML files to Redis for the AI-powered scraping system.
"""

import asyncio
import yaml
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .redis_store import RedisContextStore

logger = logging.getLogger(__name__)


@dataclass
class ScrapingConfig:
    """Represents a complete scraping configuration for a website"""
    website: str
    version: str
    last_updated: str
    status: str
    dna: Dict[str, Any]
    scraping_logic: Dict[str, Any]
    ai_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'website': self.website,
            'version': self.version,
            'last_updated': self.last_updated,
            'status': self.status,
            'dna': self.dna,
            'scraping_logic': self.scraping_logic,
            'ai_metadata': self.ai_metadata
        }


class ScrapingConfigManager:
    """
    Manages scraping configurations with YAML persistence and Redis caching
    """

    def __init__(self, config_dir: str = "scraping_configs", redis_host: str = 'localhost', redis_port: int = 6379):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.redis_store = RedisContextStore(redis_host, redis_port)
        self.loaded_configs: Dict[str, ScrapingConfig] = {}

    async def load_config_from_yaml(self, website: str) -> Optional[ScrapingConfig]:
        """
        Load a scraping configuration from YAML file

        Args:
            website: Website domain (e.g., 'govdeals.com')

        Returns:
            ScrapingConfig instance or None if not found
        """
        config_file = self.config_dir / f"{website}.yml"

        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return None

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            config = ScrapingConfig(
                website=data['website'],
                version=data['version'],
                last_updated=data['last_updated'],
                status=data.get('status', 'active'),
                dna=data.get('dna', {}),
                scraping_logic=data.get('scraping_logic', {}),
                ai_metadata=data.get('ai_metadata', {})
            )

            self.loaded_configs[website] = config
            logger.info(f"Loaded configuration for {website} v{config.version}")
            return config

        except Exception as e:
            logger.error(f"Failed to load config for {website}: {e}")
            return None

    async def save_config_to_yaml(self, config: ScrapingConfig) -> bool:
        """
        Save a scraping configuration to YAML file

        Args:
            config: ScrapingConfig instance to save

        Returns:
            True if successful, False otherwise
        """
        config_file = self.config_dir / f"{config.website}.yml"

        try:
            # Update last_updated timestamp
            config.last_updated = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

            logger.info(f"Saved configuration for {config.website} v{config.version}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config for {config.website}: {e}")
            return False

    async def sync_config_to_redis(self, config: ScrapingConfig) -> bool:
        """
        Atomically sync configuration to Redis using transactions

        Args:
            config: ScrapingConfig to sync

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use Redis pipeline for atomic updates across multiple databases
            success = await self.redis_store.atomic_config_sync(
                config.website,
                {
                    'scraping_logic': config.scraping_logic,
                    'dna': config.dna,
                    'ai_metadata': config.ai_metadata,
                    'version': config.version,
                    'last_updated': config.last_updated
                },
                ttls={
                    'scraping_logic': 604800,  # 1 week
                    'dna': 3600,              # 1 hour
                    'ai_metadata': 86400      # 24 hours
                }
            )

            if success:
                logger.info(f"Atomically synced {config.website} v{config.version} to Redis")
            else:
                logger.error(f"Failed atomic sync for {config.website}")

            return success

        except Exception as e:
            logger.error(f"Failed to sync {config.website} to Redis: {e}")
            return False

    async def get_config_from_redis(self, website: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete configuration from Redis

        Args:
            website: Website domain

        Returns:
            Complete configuration dict or None if not found
        """
        try:
            # Get scraping logic from DB 2
            scraping_logic = await self.redis_store.get_scraping_logic(website)
            if not scraping_logic:
                return None

            # Get DNA from DB 3
            dna = await self.redis_store.get_cached_dna(website)
            if not dna:
                dna = {}

            # Get AI metadata from DB 1
            ai_metadata = await self.redis_store.get_ai_knowledge(f"config:{website}")
            if not ai_metadata:
                ai_metadata = {}

            return {
                'website': website,
                'scraping_logic': scraping_logic,
                'dna': dna,
                'ai_metadata': ai_metadata,
                'source': 'redis'
            }

        except Exception as e:
            logger.error(f"Failed to get config from Redis for {website}: {e}")
            return None

    def _determine_version_bump(self, current_config: ScrapingConfig, new_logic: Dict[str, Any]) -> str:
        """
        Determine semantic version bump based on changes

        Args:
            current_config: Current configuration
            new_logic: New scraping logic

        Returns:
            Version bump type: 'major', 'minor', or 'patch'
        """
        current_logic = current_config.scraping_logic

        # Check for breaking changes (major version bump)
        current_selectors = set(current_logic.get('selectors', {}).keys())
        new_selectors = set(new_logic.get('selectors', {}).keys())

        # Removed selectors = breaking change
        if current_selectors - new_selectors:
            return 'major'

        # Changed selector values = breaking change
        for selector in current_selectors & new_selectors:
            if current_logic['selectors'][selector] != new_logic['selectors'][selector]:
                return 'major'

        # Added selectors = minor version bump
        if new_selectors - current_selectors:
            return 'minor'

        # Other logic changes (filters, pagination, etc.) = minor bump
        logic_keys = {'filters', 'pagination', 'data_mapping', 'validation_rules'}
        for key in logic_keys:
            if current_logic.get(key) != new_logic.get(key):
                return 'minor'

        # Default to patch for small changes
        return 'patch'

    async def update_config_version(self, website: str, new_logic: Dict[str, Any],
                                   change_reason: str, ai_generated: bool = False,
                                   expected_version: Optional[str] = None) -> Optional[ScrapingConfig]:
        """
        Update configuration with optimistic locking and smart semantic versioning

        Args:
            website: Website domain
            new_logic: New scraping logic
            change_reason: Reason for the change
            ai_generated: Whether this update was AI-generated
            expected_version: Expected current version for optimistic locking (None to skip check)

        Returns:
            Updated ScrapingConfig or None if failed
        """
        # Load current config
        config = await self.load_config_from_yaml(website)
        if not config:
            logger.error(f"No existing config found for {website}")
            return None

        # Optimistic locking check
        if expected_version is not None and config.version != expected_version:
            logger.error(f"Optimistic locking failed for {website}: expected {expected_version}, got {config.version}")
            return None

        # Determine version bump type
        version_bump = self._determine_version_bump(config, new_logic)

        # Calculate new version using semantic versioning
        current_parts = config.version.split('.')
        if len(current_parts) != 3:
            # Handle non-standard versions by defaulting to patch
            current_parts = ['1', '0', '0']

        major, minor, patch = map(int, current_parts)

        if version_bump == 'major':
            major += 1
            minor = 0
            patch = 0
        elif version_bump == 'minor':
            minor += 1
            patch = 0
        else:  # patch
            patch += 1

        new_version = f"{major}.{minor}.{patch}"

        # Update config
        config.version = new_version
        config.scraping_logic = new_logic
        config.last_updated = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

        # Update AI metadata
        change_record = {
            'version': new_version,
            'change_type': 'ai_adaptation' if ai_generated else 'manual_update',
            'version_bump': version_bump,
            'timestamp': config.last_updated,
            'reason': change_reason,
            'selectors_changed': self._get_selector_changes(config.scraping_logic, new_logic)
        }

        if 'adaptation_history' not in config.ai_metadata:
            config.ai_metadata['adaptation_history'] = []

        config.ai_metadata['adaptation_history'].append(change_record)
        config.ai_metadata['last_ai_analysis'] = config.last_updated
        config.ai_metadata['confidence_score'] = 0.95 if ai_generated else 1.0
        config.ai_metadata['current_version_bump_type'] = version_bump

        # Save to YAML
        if await self.save_config_to_yaml(config):
            # Sync to Redis
            if await self.sync_config_to_redis(config):
                logger.info(f"Successfully updated {website} to version {new_version} ({version_bump} bump)")
                return config

        return None

    def _get_selector_changes(self, old_logic: Dict[str, Any], new_logic: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed selector changes for history tracking"""
        old_selectors = old_logic.get('selectors', {})
        new_selectors = new_logic.get('selectors', {})

        changes = {
            'added': list(set(new_selectors.keys()) - set(old_selectors.keys())),
            'removed': list(set(old_selectors.keys()) - set(new_selectors.keys())),
            'modified': []
        }

        for selector in set(old_selectors.keys()) & set(new_selectors.keys()):
            if old_selectors[selector] != new_selectors[selector]:
                changes['modified'].append({
                    'selector': selector,
                    'old_value': old_selectors[selector],
                    'new_value': new_selectors[selector]
                })

        return changes

    async def diff_configs(self, website: str, version_a: Optional[str] = None,
                          version_b: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate detailed diff between two configuration versions

        Args:
            website: Website domain
            version_a: First version to compare (None for current)
            version_b: Second version to compare (None for current)

        Returns:
            Dictionary with detailed diff information
        """
        # Load configurations
        config_a = await self._load_config_by_version(website, version_a)
        config_b = await self._load_config_by_version(website, version_b)

        if not config_a or not config_b:
            return {'error': 'One or both versions not found'}

        return self._generate_config_diff(config_a, config_b)

    async def _load_config_by_version(self, website: str, version: Optional[str]) -> Optional[ScrapingConfig]:
        """Load configuration for a specific version"""
        if version is None:
            # Load current version
            return await self.load_config_from_yaml(website)

        # Load current config and reconstruct historical version
        current_config = await self.load_config_from_yaml(website)
        if not current_config:
            return None

        # If it's the current version
        if current_config.version == version:
            return current_config

        # Find the version in history and reconstruct
        history = current_config.ai_metadata.get('adaptation_history', [])
        target_entry = None

        for entry in history:
            if entry.get('version') == version:
                target_entry = entry
                break

        if not target_entry:
            return None

        # Reconstruct config for that version
        return await self._reconstruct_historical_config(current_config, target_entry)

    async def _reconstruct_historical_config(self, current_config: ScrapingConfig,
                                           target_entry: Dict) -> ScrapingConfig:
        """Reconstruct configuration as it was at a historical version"""
        import copy

        # This is a simplified reconstruction - in practice, you'd need
        # to store complete snapshots or reverse-apply changes
        reconstructed = copy.deepcopy(current_config)
        reconstructed.version = target_entry.get('version', 'unknown')
        reconstructed.last_updated = target_entry.get('timestamp', '')

        return reconstructed

    def _generate_config_diff(self, config_a: ScrapingConfig, config_b: ScrapingConfig) -> Dict[str, Any]:
        """Generate detailed diff between two configurations"""
        diff = {
            'versions': {
                'a': config_a.version,
                'b': config_b.version
            },
            'changes': {
                'selectors': self._diff_selectors(config_a.scraping_logic, config_b.scraping_logic),
                'filters': self._diff_dict_section(config_a.scraping_logic, config_b.scraping_logic, 'filters'),
                'pagination': self._diff_dict_section(config_a.scraping_logic, config_b.scraping_logic, 'pagination'),
                'data_mapping': self._diff_dict_section(config_a.scraping_logic, config_b.scraping_logic, 'data_mapping'),
                'rate_limiting': self._diff_dict_section(config_a.scraping_logic, config_b.scraping_logic, 'rate_limiting'),
                'error_handling': self._diff_dict_section(config_a.scraping_logic, config_b.scraping_logic, 'error_handling'),
                'dna': self._diff_dna(config_a.dna, config_b.dna)
            },
            'summary': {
                'breaking_changes': 0,
                'additions': 0,
                'removals': 0,
                'modifications': 0
            }
        }

        # Calculate summary
        for section_changes in diff['changes'].values():
            if isinstance(section_changes, dict):
                diff['summary']['breaking_changes'] += section_changes.get('breaking_changes', 0)
                diff['summary']['additions'] += len(section_changes.get('added', []))
                diff['summary']['removals'] += len(section_changes.get('removed', []))
                diff['summary']['modifications'] += len(section_changes.get('modified', []))

        return diff

    def _diff_selectors(self, logic_a: Dict, logic_b: Dict) -> Dict[str, Any]:
        """Diff selectors section with breaking change detection"""
        selectors_a = logic_a.get('selectors', {})
        selectors_b = logic_b.get('selectors', {})

        changes = {
            'added': list(set(selectors_b.keys()) - set(selectors_a.keys())),
            'removed': list(set(selectors_a.keys()) - set(selectors_b.keys())),
            'modified': [],
            'breaking_changes': 0
        }

        # Check for breaking changes (modified selectors)
        for selector in set(selectors_a.keys()) & set(selectors_b.keys()):
            if selectors_a[selector] != selectors_b[selector]:
                changes['modified'].append({
                    'selector': selector,
                    'from': selectors_a[selector],
                    'to': selectors_b[selector]
                })
                changes['breaking_changes'] += 1  # Modified selectors are breaking

        # Removed selectors are also breaking
        changes['breaking_changes'] += len(changes['removed'])

        return changes

    def _diff_dict_section(self, logic_a: Dict, logic_b: Dict, section: str) -> Dict[str, Any]:
        """Diff a dictionary section"""
        section_a = logic_a.get(section, {})
        section_b = logic_b.get(section, {})

        if section_a == section_b:
            return {'changed': False}

        return {
            'changed': True,
            'from': section_a,
            'to': section_b,
            'breaking_changes': 1 if section in ['pagination', 'data_mapping'] else 0
        }

    def _diff_dna(self, dna_a: Dict, dna_b: Dict) -> Dict[str, Any]:
        """Diff DNA section"""
        if dna_a == dna_b:
            return {'changed': False}

        changes = {'changed': True, 'differences': []}

        all_keys = set(dna_a.keys()) | set(dna_b.keys())

        for key in all_keys:
            val_a = dna_a.get(key)
            val_b = dna_b.get(key)

            if val_a != val_b:
                changes['differences'].append({
                    'field': key,
                    'from': val_a,
                    'to': val_b
                })

        return changes

    async def list_all_configs(self) -> List[str]:
        """
        List all available website configurations

        Returns:
            List of website domains with configs
        """
        configs = []
        for config_file in self.config_dir.glob("*.yml"):
            website = config_file.stem
            configs.append(website)
        return configs

    async def validate_config(self, config: ScrapingConfig) -> List[str]:
        """
        Comprehensive validation of scraping configuration with schema and business rules

        Args:
            config: Configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Schema validation
        schema_errors = self._validate_schema(config)
        errors.extend(schema_errors)

        # Business rules validation
        business_errors = self._validate_business_rules(config)
        errors.extend(business_errors)

        # Cross-reference validation
        cross_ref_errors = self._validate_cross_references(config)
        errors.extend(cross_ref_errors)

        return errors

    def _validate_schema(self, config: ScrapingConfig) -> List[str]:
        """Validate configuration against schema requirements"""
        errors = []

        # Required fields
        required_fields = ['website', 'version', 'scraping_logic']
        for field in required_fields:
            if not getattr(config, field, None):
                errors.append(f"Missing required field: {field}")

        # Version format validation
        if config.version:
            if not self._is_valid_semantic_version(config.version):
                errors.append(f"Invalid version format: {config.version}. Must be semantic version (x.y.z)")

        # Website format validation
        if config.website:
            if not self._is_valid_domain(config.website):
                errors.append(f"Invalid website domain: {config.website}")

        # Scraping logic structure validation
        logic = config.scraping_logic
        if not isinstance(logic, dict):
            errors.append("scraping_logic must be a dictionary")
        else:
            if 'selectors' not in logic:
                errors.append("scraping_logic must contain 'selectors'")
            elif not isinstance(logic['selectors'], dict):
                errors.append("'selectors' must be a dictionary")
            else:
                # Validate selector values are strings
                for key, value in logic['selectors'].items():
                    if not isinstance(value, str):
                        errors.append(f"Selector '{key}' must be a string, got {type(value)}")

            # Validate optional sections
            if 'filters' in logic and not isinstance(logic['filters'], dict):
                errors.append("'filters' must be a dictionary")

            if 'pagination' in logic and not isinstance(logic['pagination'], dict):
                errors.append("'pagination' must be a dictionary")

            if 'data_mapping' in logic and not isinstance(logic['data_mapping'], dict):
                errors.append("'data_mapping' must be a dictionary")

        # DNA structure validation
        dna = config.dna
        if dna:
            if not isinstance(dna, dict):
                errors.append("DNA must be a dictionary")
            elif 'framework' not in dna:
                errors.append("DNA should contain 'framework' field")

        # AI metadata validation
        ai_meta = config.ai_metadata
        if ai_meta and not isinstance(ai_meta, dict):
            errors.append("ai_metadata must be a dictionary")

        return errors

    def _validate_business_rules(self, config: ScrapingConfig) -> List[str]:
        """Validate business rules and constraints"""
        errors = []

        logic = config.scraping_logic

        # At least one selector required
        if 'selectors' in logic and len(logic['selectors']) == 0:
            errors.append("At least one selector must be defined")

        # Rate limiting validation
        if 'rate_limiting' in logic:
            rl = logic['rate_limiting']
            if 'requests_per_minute' in rl:
                rpm = rl['requests_per_minute']
                if not isinstance(rpm, int) or rpm <= 0:
                    errors.append("requests_per_minute must be a positive integer")
                elif rpm > 120:  # Reasonable upper limit
                    errors.append("requests_per_minute cannot exceed 120")

        # Timeout validation
        if 'error_handling' in logic:
            eh = logic['error_handling']
            if 'timeout_seconds' in eh:
                timeout = eh['timeout_seconds']
                if not isinstance(timeout, int) or timeout <= 0:
                    errors.append("timeout_seconds must be a positive integer")
                elif timeout > 300:  # 5 minutes max
                    errors.append("timeout_seconds cannot exceed 300")

        # Version consistency check
        if config.ai_metadata and 'adaptation_history' in config.ai_metadata:
            history = config.ai_metadata['adaptation_history']
            if history:
                # Check that versions are monotonically increasing
                versions = [entry.get('version', '') for entry in history]
                versions.append(config.version)
                if not self._are_versions_monotonic(versions):
                    errors.append("Version history is not monotonically increasing")

        return errors

    def _validate_cross_references(self, config: ScrapingConfig) -> List[str]:
        """Validate cross-references between different config sections"""
        errors = []

        logic = config.scraping_logic
        dna = config.dna

        # Check if DNA framework matches expected patterns
        if dna and 'framework' in dna and logic and 'selectors' in logic:
            framework = dna['framework'].lower()
            selectors = logic['selectors']

            # Framework-specific validations
            if framework == 'react':
                # React apps often use data-* attributes
                has_data_attrs = any('data-' in str(sel) for sel in selectors.values())
                if not has_data_attrs:
                    errors.append("React framework detected but no data-* attributes in selectors")

            elif framework == 'angular':
                # Angular often uses ng-* or [attr]
                has_ng_attrs = any('ng-' in str(sel) or '[' in str(sel) for sel in selectors.values())
                if not has_ng_attrs:
                    errors.append("Angular framework detected but no ng-* or [attr] selectors found")

        # Validate data mapping references selectors
        if 'data_mapping' in logic:
            mapping = logic['data_mapping']
            selectors = logic.get('selectors', {})

            for field, selector_ref in mapping.items():
                if selector_ref not in selectors:
                    errors.append(f"data_mapping field '{field}' references unknown selector '{selector_ref}'")

        return errors

    def _is_valid_semantic_version(self, version: str) -> bool:
        """Check if version string is valid semantic version (x.y.z)"""
        import re
        pattern = r'^\d+\.\d+\.\d+$'
        return bool(re.match(pattern, version))

    def _is_valid_domain(self, domain: str) -> bool:
        """Basic domain validation"""
        import re
        # Simple domain pattern (not comprehensive)
        pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        return bool(re.match(pattern, domain)) and len(domain) <= 253

    def _are_versions_monotonic(self, versions: List[str]) -> bool:
        """Check if versions are in monotonically increasing order"""
        def version_tuple(v):
            try:
                return tuple(map(int, v.split('.')))
            except:
                return (0, 0, 0)

        sorted_versions = sorted(versions, key=version_tuple)
        return versions == sorted_versions

    async def initialize_from_yaml_files(self) -> Dict[str, bool]:
        """
        Load all YAML configuration files and sync to Redis

        Returns:
            Dict mapping website to success status
        """
        results = {}
        config_files = list(self.config_dir.glob("*.yml"))

        for config_file in config_files:
            website = config_file.stem

            # Load from YAML
            config = await self.load_config_from_yaml(website)
            if not config:
                results[website] = False
                continue

            # Validate
            validation_errors = await self.validate_config(config)
            if validation_errors:
                logger.error(f"Validation failed for {website}: {validation_errors}")
                results[website] = False
                continue

            # Sync to Redis
            success = await self.sync_config_to_redis(config)
            results[website] = success

        return results

    async def rollback_config(self, website: str, target_version: str) -> Optional[ScrapingConfig]:
        """
        Rollback configuration to a previous version

        Args:
            website: Website domain
            target_version: Version to rollback to

        Returns:
            Updated ScrapingConfig with rolled back version, or None if failed
        """
        # Load current config
        current_config = await self.load_config_from_yaml(website)
        if not current_config:
            logger.error(f"No current config found for {website}")
            return None

        # Find target version in history
        history = current_config.ai_metadata.get('adaptation_history', [])
        target_entry = None

        for entry in history:
            if entry.get('version') == target_version:
                target_entry = entry
                break

        if not target_entry:
            logger.error(f"Target version {target_version} not found in history for {website}")
            return None

        # Check if we're rolling back to a newer version (not allowed)
        current_tuple = self._version_tuple(current_config.version)
        target_tuple = self._version_tuple(target_version)

        if target_tuple > current_tuple:
            logger.error(f"Cannot rollback to newer version {target_version} from {current_config.version}")
            return None

        # Create rollback config by reverting changes
        rollback_config = await self._create_rollback_config(current_config, target_entry)

        # Validate rollback config
        validation_errors = await self.validate_config(rollback_config)
        if validation_errors:
            logger.error(f"Rollback config validation failed: {validation_errors}")
            return None

        # Save rollback config
        rollback_config.version = target_version
        rollback_config.last_updated = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

        # Update AI metadata for rollback
        rollback_record = {
            'version': target_version,
            'change_type': 'rollback',
            'timestamp': rollback_config.last_updated,
            'reason': f'Rolled back from {current_config.version} to {target_version}',
            'previous_version': current_config.version
        }

        if 'adaptation_history' not in rollback_config.ai_metadata:
            rollback_config.ai_metadata['adaptation_history'] = []

        rollback_config.ai_metadata['adaptation_history'].append(rollback_record)
        rollback_config.ai_metadata['last_rollback'] = rollback_config.last_updated

        # Save to YAML
        if await self.save_config_to_yaml(rollback_config):
            # Sync to Redis
            if await self.sync_config_to_redis(rollback_config):
                logger.info(f"Successfully rolled back {website} to version {target_version}")
                return rollback_config

        return None

    async def _create_rollback_config(self, current_config: ScrapingConfig, target_entry: Dict) -> ScrapingConfig:
        """
        Create configuration for rollback by reverting changes

        Args:
            current_config: Current configuration
            target_entry: History entry for target version

        Returns:
            ScrapingConfig for the target version
        """
        import copy

        # Start with current config as base
        rollback_config = copy.deepcopy(current_config)

        # Get all history entries after target version
        history = current_config.ai_metadata.get('adaptation_history', [])
        target_index = None

        for i, entry in enumerate(history):
            if entry.get('version') == target_entry.get('version'):
                target_index = i
                break

        if target_index is None:
            return rollback_config

        # Get changes made after target version
        subsequent_changes = history[target_index + 1:]

        # Reverse apply changes to get back to target state
        for change in reversed(subsequent_changes):
            if change.get('change_type') == 'ai_adaptation':
                selectors_changed = change.get('selectors_changed', {})

                # Revert selector changes
                for change_type, changes in selectors_changed.items():
                    if change_type == 'added':
                        # Remove added selectors
                        for selector in changes:
                            rollback_config.scraping_logic['selectors'].pop(selector, None)
                    elif change_type == 'removed':
                        # This is complex - we'd need to know what was removed
                        # For now, skip as we don't store the removed values
                        pass
                    elif change_type == 'modified':
                        # Revert modified selectors
                        for change_info in changes:
                            selector = change_info['selector']
                            old_value = change_info['old_value']
                            rollback_config.scraping_logic['selectors'][selector] = old_value

        return rollback_config

    async def get_current_version(self, website: str) -> Optional[str]:
        """
        Get the current version of a website configuration

        Args:
            website: Website domain

        Returns:
            Current version string or None if not found
        """
        config = await self.load_config_from_yaml(website)
        return config.version if config else None

    async def list_config_versions(self, website: str) -> List[Dict[str, Any]]:
        """
        List all available versions for a website configuration

        Args:
            website: Website domain

        Returns:
            List of version info dictionaries
        """
        config = await self.load_config_from_yaml(website)
        if not config:
            return []

        versions = [{
            'version': config.version,
            'last_updated': config.last_updated,
            'status': config.status,
            'is_current': True
        }]

        # Add historical versions
        history = config.ai_metadata.get('adaptation_history', [])
        for entry in history:
            versions.append({
                'version': entry.get('version'),
                'last_updated': entry.get('timestamp'),
                'change_type': entry.get('change_type'),
                'reason': entry.get('reason'),
                'is_current': False
            })

        # Sort by version descending
        versions.sort(key=lambda x: self._version_tuple(x['version']), reverse=True)

        return versions

    def _version_tuple(self, version: str) -> tuple:
        """Convert version string to tuple for comparison"""
        try:
            return tuple(map(int, version.split('.')))
        except:
            return (0, 0, 0)

    def close(self):
        """Clean up resources"""
        self.redis_store.close()