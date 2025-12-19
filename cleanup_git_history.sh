#!/bin/bash
# QUICK GIT CLEANUP với git filter-repo

echo "🧹 Cleaning git repo..."

# 1. Check current size
echo "Current size:"
git count-objects -vH | grep size-pack

# 2. Install git-filter-repo (nếu chưa có)
# pip install git-filter-repo

# 3. Remove large folders from history
echo ""
echo "Removing large folders from history..."
git filter-repo --path datasets/raw --invert-paths --force
git filter-repo --path datasets/processed --invert-paths --force
git filter-repo --path datasets/splits --invert-paths --force
git filter-repo --path models/artifacts --invert-paths --force
git filter-repo --path results --invert-paths --force

# 4. Cleanup
echo ""
echo "Garbage collecting..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 5. Check new size
echo ""
echo "New size:"
git count-objects -vH | grep size-pack

echo ""
echo "✅ Done! Repo cleaned."
echo "Now run: git push -f origin main"
