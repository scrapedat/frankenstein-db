#!/bin/bash
# FrankensteinDB Production Deployment Script
# Quick deployment for production environments

set -e

echo "🧟‍♂️ FrankensteinDB Production Deployment"
echo "=========================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  Warning: Running as root is not recommended for production"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Setup production environment
echo "🔧 Setting up production environment..."
./setup-production.sh

# Build production image
echo "🏗️ Building production Docker image..."
docker-compose -f docker-compose.production.yml build

# Start the system
echo "🚀 Starting FrankensteinDB production system..."
docker-compose -f docker-compose.production.yml up -d

# Wait a moment for startup
echo "⏳ Waiting for services to start..."
sleep 10

# Check system status
echo "📊 Checking system status..."
./scripts/monitor-frankenstein.sh

echo ""
echo "✅ FrankensteinDB Production Deployment Complete!"
echo ""
echo "🔗 Access Points:"
echo "  • Redis Commander: http://localhost:8081 (if debug profile enabled)"
echo "  • Logs: docker-compose -f docker-compose.production.yml logs -f"
echo ""
echo "📚 Management Commands:"
echo "  • Monitor: ./scripts/monitor-frankenstein.sh"
echo "  • Backup:  ./scripts/backup-frankenstein.sh"
echo "  • Stop:    docker-compose -f docker-compose.production.yml down"
echo ""
echo "📖 Integration Example:"
echo "from src.production_config import get_production_frankenstein"
echo "frankenstein = get_production_frankenstein()"
echo ""