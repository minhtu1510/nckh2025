#!/usr/bin/env python3
"""
Quick fix script để chạy experiments
Tự động setup PYTHONPATH
"""
import sys
from pathlib import Path

# Add parent directory to path để import ids_research
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

# Now run main
if __name__ == "__main__":
    from ids_research import main
    main.main()
