#!/usr/bin/env python3
"""
Test script for the scraping configuration system
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.scraping_config_manager import ScrapingConfigManager


async def test_config_system():
    """Test the scraping configuration system"""
    print("🧪 Testing Scraping Configuration System...")

    # Initialize manager
    manager = ScrapingConfigManager()

    try:
        # Test 1: Load configuration from YAML
        print("\n1. Loading configuration from YAML...")
        config = await manager.load_config_from_yaml('govdeals.com')
        if config:
            print(f"✅ Loaded config for {config.website} v{config.version}")
            print(f"   Status: {config.status}")
            print(f"   Framework: {config.dna.get('framework', 'unknown')}")
            print(f"   Selectors: {len(config.scraping_logic.get('selectors', {}))}")
        else:
            print("❌ Failed to load config")
            return False

        # Test 2: Validate configuration
        print("\n2. Validating configuration...")
        errors = await manager.validate_config(config)
        if errors:
            print(f"❌ Validation errors: {errors}")
            return False
        else:
            print("✅ Configuration is valid")

        # Test 3: Sync to Redis
        print("\n3. Syncing configuration to Redis...")
        success = await manager.sync_config_to_redis(config)
        if success:
            print("✅ Successfully synced to Redis")
        else:
            print("❌ Failed to sync to Redis")
            return False

        # Test 4: Retrieve from Redis
        print("\n4. Retrieving configuration from Redis...")
        redis_config = await manager.get_config_from_redis('govdeals.com')
        if redis_config:
            print("✅ Retrieved config from Redis")
            print(f"   Source: {redis_config.get('source')}")
            print(f"   Has scraping logic: {'scraping_logic' in redis_config}")
            print(f"   Has DNA: {'dna' in redis_config}")
        else:
            print("❌ Failed to retrieve from Redis")
            return False

        # Test 5: Update configuration version
        print("\n5. Testing configuration update...")
        new_logic = config.scraping_logic.copy()
        new_logic['selectors'] = {**new_logic['selectors'], 'test_selector': '.test-class'}

        updated_config = await manager.update_config_version(
            'govdeals.com',
            new_logic,
            'Added test selector for validation',
            ai_generated=True
        )

        if updated_config:
            print(f"✅ Updated config to version {updated_config.version}")
            print(f"   AI generated: True")
            print(f"   Change history: {len(updated_config.ai_metadata.get('adaptation_history', []))} entries")
        else:
            print("❌ Failed to update config")
            return False

        # Test 6: List all configurations
        print("\n6. Listing all configurations...")
        configs = await manager.list_all_configs()
        print(f"✅ Found {len(configs)} configuration files: {configs}")

        print("\n🎉 All tests passed! Scraping configuration system is working.")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        manager.close()


if __name__ == "__main__":
    success = asyncio.run(test_config_system())
    sys.exit(0 if success else 1)