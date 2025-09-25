#!/bin/bash
# Sync FrankensteinDB changes to production-vms monorepo

set -e

echo "🔄 Syncing FrankensteinDB to production-vms monorepo..."

# Check if we're in the right directory
if [ ! -f "src/frankenstein_db.py" ]; then
    echo "❌ Error: Not in frankenstein-db directory"
    exit 1
fi

# Check if monorepo exists (assuming it's a sibling directory)
MONOREPO_DIR="../production-vms"
if [ ! -d "$MONOREPO_DIR" ]; then
    echo "❌ Error: Monorepo directory not found at $MONOREPO_DIR"
    echo "Please clone the monorepo: git clone https://github.com/scrapedat/production-vms.git"
    exit 1
fi

# Create frankenstein-db directory in monorepo if it doesn't exist
MONOREPO_DB_DIR="$MONOREPO_DIR/frankenstein-db"
mkdir -p "$MONOREPO_DB_DIR"

echo "📁 Copying files to monorepo..."

# Copy all files except .git and venv
rsync -av --exclude='.git/' --exclude='venv/' --exclude='__pycache__/' \
      --exclude='*.pyc' --exclude='.pytest_cache/' \
      ./ "$MONOREPO_DB_DIR/"

echo "✅ Files copied successfully"

# Check if there are changes in the monorepo
cd "$MONOREPO_DIR"
if git diff --quiet --exit-code; then
    echo "ℹ️  No changes detected in monorepo"
else
    echo "📝 Changes detected. Here's what changed:"
    git status --porcelain

    echo ""
    echo "🚀 Ready to commit and push to monorepo!"
    echo "Run these commands in $MONOREPO_DIR:"
    echo "  git add frankenstein-db/"
    echo "  git commit -m 'feat: Update frankenstein-db with AI-powered scraping system'"
    echo "  git push origin main"
fi

echo "🎉 Sync complete!"