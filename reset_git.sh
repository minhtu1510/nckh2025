#!/bin/bash
# Backup current code
echo "Creating backup..."
mkdir -p /tmp/ids_backup
cp -r *.py *.md configs pipelines attacks models/baselines datasets/splits/cicids2017/metadata.json /tmp/ids_backup/ 2>/dev/null || true

# Remove git
echo "Removing old git..."
rm -rf .git

# New repo
echo "Creating fresh repo..."
git init
git add .
git commit -m "Clean repo - code only"

echo "✅ Done! Now run:"
echo "git remote add origin YOUR_REPO_URL"
echo "git push -f origin main"
