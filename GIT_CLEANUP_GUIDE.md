# GIT CLEANUP - XÓA LARGE FILES

## Vấn đề: Git repo 1.16GB (quá nặng!)

### Large files trong history:
```
2.9GB - ids_research/datasets/raw/CIC-ToN-IoT.csv
580MB - ids_research/datasets/raw/CIC-ToN-IoT.zip
545MB - ids_research/datasets/processed/cicids2018_processed.csv
440MB - ids_research/datasets/raw/CIC-ToN-IoT-V2.parquet
358MB - ids_research/datasets/raw/cicids2018.csv
189MB - ids_research/datasets/splits/exp1_baseline/X_train.npy (x2)
... và nhiều files khác
```

## ✅ GIẢI PHÁP:

### Option 1: BFG Repo-Cleaner (RECOMMEND)
```bash
# Install BFG
# Ubuntu: sudo apt install bfg
# Or download from: https://rtyley.github.io/bfg-repo-cleaner/

# Clean files > 10MB
bfg --strip-blobs-bigger-than 10M

# Clean specific folders
bfg --delete-folders datasets
bfg --delete-folders models/artifacts

# Cleanup
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Option 2: Git Filter-Repo (Better)
```bash
# Install
pip install git-filter-repo

# Remove large files/folders
git filter-repo --path-glob 'datasets/raw/*' --invert-paths
git filter-repo --path-glob 'datasets/processed/*' --invert-paths  
git filter-repo --path-glob 'datasets/splits/*' --invert-paths
git filter-repo --path-glob 'models/artifacts/*.pkl' --invert-paths
git filter-repo --path-glob 'models/artifacts/*.h5' --invert-paths
```

### Option 3: Start Fresh (EASIEST)
```bash
# Backup code only
mkdir ../ids_research_backup
cp -r *.py ../ids_research_backup/
cp -r configs ../ids_research_backup/
cp -r pipelines ../ids_research_backup/
cp -r attacks ../ids_research_backup/
cp -r models ../ids_research_backup/
cp .gitignore ../ids_research_backup/

# Remove git history
rm -rf .git

# Init new repo
git init
git add .
git commit -m "Clean repo - code only"

# Force push (WARNING: overwrites remote)
git remote add origin <your-repo-url>
git push -f origin main
```

## 🎯 QUICK FIX - Keep History

If you want to keep history but clean up:

```bash
# 1. Remove files from future commits (already done via .gitignore)
git add .gitignore
git commit -m "Update gitignore"

# 2. Clean git cache
git rm -r --cached datasets/
git rm -r --cached models/artifacts/*.pkl
git commit -m "Remove large files from tracking"

# 3. Garbage collect
git gc --aggressive --prune=now

# 4. Push
git push origin main
```

**Note**: This doesn't remove from history, just from future commits.

## 💾 DATA BACKUP

Keep data locally, don't push:
```
datasets/          → In .gitignore ✅
models/artifacts/  → In .gitignore ✅
results/**/*.npy   → In .gitignore ✅
```

Only push:
- Code (*.py)
- Configs (*.yaml, *.json small files)
- Documentation (*.md)
- Small results (*.csv metrics)

Target repo size: < 50MB
