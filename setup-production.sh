#!/bin/bash
# FrankensteinDB Production Setup Script

set -e

echo "🧟‍♂️ Setting up FrankensteinDB for Production..."

# Create production data directories
echo "📁 Creating production data directories..."
mkdir -p production-data/{sqlite,blobs,redis,logs,backups}

# Set appropriate permissions
echo "🔒 Setting permissions..."
chmod 755 production-data
chmod 700 production-data/sqlite
chmod 700 production-data/blobs
chmod 700 production-data/redis
chmod 755 production-data/logs
chmod 700 production-data/backups

# Create environment file
echo "⚙️ Creating production environment..."
cat > .env.production << EOF
# FrankensteinDB Production Configuration
COMPOSE_PROJECT_NAME=frankenstein-production
FRANKENSTEIN_LOG_LEVEL=INFO
FRANKENSTEIN_REDIS_HOST=redis
FRANKENSTEIN_REDIS_PORT=6379

# Data persistence paths
FRANKENSTEIN_EVOLUTION_DB=/data/sqlite/website_evolution.db
FRANKENSTEIN_SEARCH_DB=/data/sqlite/search_index.db
FRANKENSTEIN_BLOB_STORAGE=/data/blobs

# Backup configuration
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
EOF

# Create backup script
echo "💾 Creating backup script..."
cat > scripts/backup-frankenstein.sh << 'EOF'
#!/bin/bash
# FrankensteinDB Backup Script

set -e

BACKUP_DIR="./production-data/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="frankenstein_backup_${TIMESTAMP}"

echo "🔄 Starting FrankensteinDB backup: ${BACKUP_NAME}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup SQLite databases
echo "📊 Backing up SQLite databases..."
cp production-data/sqlite/*.db "${BACKUP_DIR}/${BACKUP_NAME}/" 2>/dev/null || echo "No SQLite databases found"

# Backup blob storage (compressed)
echo "💾 Backing up blob storage..."
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/blob_storage.tar.gz" production-data/blobs/ 2>/dev/null || echo "No blob data found"

# Backup Redis data
echo "🔴 Backing up Redis data..."
docker-compose -f docker-compose.production.yml exec -T redis redis-cli BGSAVE
sleep 5  # Wait for background save
cp production-data/redis/dump.rdb "${BACKUP_DIR}/${BACKUP_NAME}/" 2>/dev/null || echo "No Redis dump found"

# Create backup manifest
cat > "${BACKUP_DIR}/${BACKUP_NAME}/manifest.json" << MANIFEST
{
  "backup_name": "${BACKUP_NAME}",
  "timestamp": "$(date -Iseconds)",
  "components": [
    "sqlite_databases",
    "blob_storage", 
    "redis_data"
  ],
  "size_mb": $(du -sm "${BACKUP_DIR}/${BACKUP_NAME}" | cut -f1)
}
MANIFEST

echo "✅ Backup completed: ${BACKUP_DIR}/${BACKUP_NAME}"

# Cleanup old backups (keep last 30 days)
find "${BACKUP_DIR}" -name "frankenstein_backup_*" -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true

echo "🧹 Old backups cleaned up"
EOF

# Create scripts directory and make backup script executable
mkdir -p scripts
chmod +x scripts/backup-frankenstein.sh

# Create monitoring script
echo "📊 Creating monitoring script..."
cat > scripts/monitor-frankenstein.sh << 'EOF'
#!/bin/bash
# FrankensteinDB Monitoring Script

echo "🧟‍♂️ FrankensteinDB Production Status"
echo "======================================"

# Check container status
echo "📦 Container Status:"
docker-compose -f docker-compose.production.yml ps

echo ""
echo "💾 Storage Usage:"
du -sh production-data/*

echo ""
echo "🔴 Redis Status:"
docker-compose -f docker-compose.production.yml exec redis redis-cli info replication | head -5

echo ""
echo "📊 Database Sizes:"
ls -lh production-data/sqlite/*.db 2>/dev/null || echo "No SQLite databases found"

echo ""
echo "🗂️ Blob Storage Summary:"
find production-data/blobs -type f | wc -l | awk '{print $1 " blob files"}'

echo ""
echo "📈 Recent Activity:"
tail -5 production-data/logs/*.log 2>/dev/null || echo "No log files found"
EOF

chmod +x scripts/monitor-frankenstein.sh

# Create restoration script
echo "🔄 Creating restoration script..."
cat > scripts/restore-frankenstein.sh << 'EOF'
#!/bin/bash
# FrankensteinDB Restoration Script

if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup_name>"
    echo "Available backups:"
    ls -1 production-data/backups/ | grep frankenstein_backup_
    exit 1
fi

BACKUP_NAME=$1
BACKUP_PATH="production-data/backups/${BACKUP_NAME}"

if [ ! -d "${BACKUP_PATH}" ]; then
    echo "❌ Backup not found: ${BACKUP_PATH}"
    exit 1
fi

echo "🔄 Restoring FrankensteinDB from backup: ${BACKUP_NAME}"

# Stop services
docker-compose -f docker-compose.production.yml down

# Restore SQLite databases
echo "📊 Restoring SQLite databases..."
cp "${BACKUP_PATH}"/*.db production-data/sqlite/ 2>/dev/null || echo "No SQLite databases in backup"

# Restore blob storage
echo "💾 Restoring blob storage..."
if [ -f "${BACKUP_PATH}/blob_storage.tar.gz" ]; then
    rm -rf production-data/blobs/*
    tar -xzf "${BACKUP_PATH}/blob_storage.tar.gz" -C production-data/blobs/ --strip-components=2
fi

# Restore Redis data
echo "🔴 Restoring Redis data..."
if [ -f "${BACKUP_PATH}/dump.rdb" ]; then
    cp "${BACKUP_PATH}/dump.rdb" production-data/redis/
fi

# Start services
docker-compose -f docker-compose.production.yml up -d

echo "✅ Restoration completed from backup: ${BACKUP_NAME}"
EOF

chmod +x scripts/restore-frankenstein.sh

echo ""
echo "✅ FrankensteinDB Production Setup Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Review and customize .env.production"
echo "2. Start the system: docker-compose -f docker-compose.production.yml up -d"
echo "3. Monitor: ./scripts/monitor-frankenstein.sh"
echo "4. Backup: ./scripts/backup-frankenstein.sh"
echo ""
echo "🔧 Management Commands:"
echo "  • Start:    docker-compose -f docker-compose.production.yml up -d"
echo "  • Stop:     docker-compose -f docker-compose.production.yml down"
echo "  • Logs:     docker-compose -f docker-compose.production.yml logs -f"
echo "  • Monitor:  ./scripts/monitor-frankenstein.sh"
echo "  • Backup:   ./scripts/backup-frankenstein.sh"
echo ""