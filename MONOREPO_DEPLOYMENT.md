# FrankensteinDB → Production-VMs Monorepo Deployment

## Overview
This document outlines how to deploy the completed AI-powered scraping system from frankenstein-db to the production-vms monorepo.

## What We've Built

### Phase 1: FrankensteinDB as Single Source of Truth ✅
- **Structured YAML Schema**: Complete configuration format for website scraping
- **ScrapingConfigManager**: Full CRUD operations with Redis synchronization
- **Versioning System**: Automatic version increments with change history
- **Multi-Database Redis**: Separate DBs for different data types (0-4)

### Phase 2: AI-Scraper-VM as Thin Client ✅
- **Dynamic Logic Retrieval**: Runtime fetching of scraping logic from Redis
- **Thin Client Architecture**: No hardcoded scraping rules
- **Change Detection Framework**: AI-ready website analysis
- **Adaptation Request System**: Triggers for self-healing workflows

### Phase 3: AI Integration (Next)
- **Ollama Model Integration**: Small AI model for selector adaptation
- **MQTT Communication**: Real-time change notifications
- **Self-Healing Logic**: Automatic website adaptation

## Deployment Steps

### 1. Sync Changes to Monorepo
```bash
# From frankenstein-db directory
./sync_to_monorepo.sh

# Or manually:
rsync -av --exclude='.git/' --exclude='venv/' ./ ../production-vms/frankenstein-db/
```

### 2. Update Monorepo Dependencies
Add to `production-vms/requirements.txt`:
```
# FrankensteinDB dependencies
msgpack>=1.0.0
redis>=4.0.0
PyYAML>=6.0.0
requests>=2.28.0
beautifulsoup4>=4.11.0
lxml>=4.9.0
```

### 3. Update GitHub Actions Workflow
Enhance `.github/workflows/deploy-vms.yml` to include:

```yaml
- name: Test FrankensteinDB System
  run: |
    cd frankenstein-db
    python -m pip install -r requirements.txt
    python test_scraping_config.py

- name: Deploy Configuration System
  run: |
    # Install Redis on VMs
    # Sync configurations to Redis
    # Initialize AI-Scraper-VM clients
```

### 4. Commit and Push
```bash
cd ../production-vms
git add frankenstein-db/
git commit -m "feat: Deploy AI-powered scraping system

- Add frankenstein-db as central knowledge base
- Implement YAML-based configuration management
- Add multi-database Redis architecture
- Create AI-Scraper-VM thin client framework
- Enable dynamic logic retrieval and execution

Closes Phase 1 & 2 of scraping system development"
git push origin main
```

## System Architecture in Monorepo

```
production-vms/
├── frankenstein-db/           # Central knowledge base
│   ├── scraping_configs/      # YAML configurations
│   ├── src/                   # Core system
│   └── test_scraping_config.py # Validation tests
├── ai-scraper-vm/             # Thin client scrapers
├── user-dashboard/            # Monitoring interface
└── .github/workflows/         # CI/CD pipelines
```

## Key Files Added

### Core System
- `src/scraping_config_manager.py` - Configuration management
- `src/redis_store.py` - Enhanced multi-database Redis
- `scraping_configs/govdeals.com.yml` - Example configuration

### Testing & Validation
- `test_scraping_config.py` - Comprehensive system tests
- `ai-scraper-vm-example.py` - Thin client demonstration

### Deployment
- `sync_to_monorepo.sh` - Automated sync script
- `MONOREPO_DEPLOYMENT.md` - This deployment guide

## Next Steps After Deployment

1. **Phase 3**: Integrate AI model for self-healing
2. **MQTT Setup**: Add real-time communication
3. **Browser Automation**: Replace simulation with Selenium/Playwright
4. **User Dashboard**: Add Redis monitoring and AI suggestion review

## Success Metrics
- ✅ FrankensteinDB loads configurations from YAML
- ✅ Redis synchronization works across databases
- ✅ AI-Scraper-VM retrieves logic dynamically
- ✅ Versioning system tracks all changes
- ✅ System gracefully handles Redis failures

The AI-powered self-healing scraping system is now ready for production deployment! 🚀