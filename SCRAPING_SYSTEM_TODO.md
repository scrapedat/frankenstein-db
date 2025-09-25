# AI-Powered Self-Healing Web Scraper - Implementation Roadmap

## System Architecture Overview
Transforming frankenstein-db, ai-scraper-vm, and user-dashboard into a cohesive AI-powered scraping platform with self-healing capabilities.

## Phase 1: Foundation (High Priority)

### 1. Refactor frankenstein-db as single source of truth for scraping logic and website DNA
- Treat frankenstein-db as the central knowledge base
- Store all scraping logic and website configurations in structured format
- Enable Redis synchronization for fast access

### 2. Implement structured YAML schema in frankenstein-db for website configurations
- Create consistent YAML structure for each website
- Include fields: URL, version, website DNA, scraping logic components
- Make data both human-readable and machine-parsable

### 3. Add versioning system to YAML files and Redis synchronization
- Use Git commits for change tracking
- Add version numbers within YAML files (e.g., version: 1.2.0)
- Create Redis-sync process to load YAML data into Redis automatically

### 4. Update ai-scraper-vm to be thin client that retrieves logic from Redis
- Remove hardcoded scraping logic from VMs
- Make VMs dynamic clients that query Redis for instructions
- Enable runtime logic retrieval before scraping jobs

## Phase 2: AI Integration (High Priority)

### 5. Implement Redis client in ai-scraper-vm for dynamic logic retrieval
- Add Redis client library to scraper VMs
- Query Redis using website URL to get latest scraping logic
- Handle connection failures gracefully

### 6. Build dynamic logic execution engine in ai-scraper-vm
- Refactor VM code to execute logic from Redis dynamically
- Read CSS selectors and patterns from Redis data at runtime
- Apply retrieved logic without prior knowledge of structure

### 7. Integrate AI model into ai-scraper-vm for self-healing scraping
- Add small Ollama model to scraper VMs
- Enable AI to listen for website change notifications
- Use frankenstein-db data to generate updated scraping logic

## Phase 3: Human-in-the-Loop (High Priority)

### 8. Refactor user-dashboard for real-time Redis monitoring and AI suggestions
- Connect dashboard to Redis for live status display
- Show current scraping logic versions for each website
- Display real-time system health and performance metrics

### 9. Add manual override interface in user-dashboard for developer intervention
- Create interface for reviewing AI-generated logic suggestions
- Enable approve/reject workflow for AI proposals
- Provide manual entry interface for complex scraping logic updates

## Phase 4: Advanced Features (Medium Priority)

### 10. Implement MQTT broker for real-time communication between components
- Set up MQTT broker as central nervous system
- Create topics: website/changes, ai/suggestions, logic/updates
- Enable low-latency communication between all components

### 11. Build website DNA collector (browser extension or monitoring script)
- Create component to detect website structural changes
- Capture new website DNA when changes are detected
- Publish change notifications to MQTT topics

### 12. Create AI tools for find_logic, update_logic, and publish_to_mqtt
- Define tools.yml with function definitions and parameters
- Implement find_logic: query Redis for latest scraping logic
- Implement update_logic: update Redis and trigger YAML commits
- Implement publish_to_mqtt: send real-time notifications

### 13. Set up GitHub Actions for AI-driven YAML updates in frankenstein-db
- Create GitHub Action triggered by AI service
- Authenticate with GitHub API for YAML file updates
- Automate commits of validated scraping logic changes

## Technical Implementation Details

### YAML Schema Example
```yaml
govdeals.com:
  version: "1.2.0"
  last_updated: "2024-01-15T10:30:00Z"
  dna:
    framework: "react"
    selectors:
      title: ".auction-title"
      price: ".current-bid"
      ending: ".time-remaining"
  scraping_logic:
    steps:
      - navigate: "https://govdeals.com"
      - wait_for: ".auction-list"
      - extract:
          auctions: ".auction-item"
          filters: "ending_in_48h"
```

### Redis Multi-Database Usage
- **DB 0**: User contexts and sessions
- **DB 1**: AI-learned scraping patterns and rules
- **DB 2**: Website-specific scraping logic and configurations
- **DB 3**: Cached website DNA and analysis results
- **DB 4**: Session metadata and AI suggestions

## Success Metrics
- 95%+ automated adaptation to website changes
- <5% manual intervention required
- Real-time scraping logic updates
- Complete data privacy on local VMs
- Cost-effective scaling vs cloud services