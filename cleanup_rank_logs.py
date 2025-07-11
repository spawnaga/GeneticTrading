
#!/usr/bin/env python
"""
Clean up unnecessary rank-specific log folders
"""

import os
import shutil
from pathlib import Path

def cleanup_rank_logs(logs_dir="./logs"):
    """Remove rank-specific log folders that are no longer needed."""
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        print("No logs directory found.")
        return
    
    removed_count = 0
    
    # Find and remove rank-specific folders
    for item in logs_path.iterdir():
        if item.is_dir() and (
            item.name.startswith("train_rank") or 
            item.name.startswith("test_rank")
        ):
            print(f"Removing: {item}")
            shutil.rmtree(item)
            removed_count += 1
    
    # Also clean up any rank-specific log files
    for item in logs_path.iterdir():
        if item.is_file() and "rank" in item.name:
            file_size = item.stat().st_size
            if file_size == 0 or item.suffix in ['.log', '.txt']:
                print(f"Removing empty/small rank file: {item}")
                item.unlink()
                removed_count += 1
    
    print(f"Cleanup complete. Removed {removed_count} items.")
    
    # Show remaining structure
    print("\nRemaining log structure:")
    for item in sorted(logs_path.iterdir()):
        if item.is_dir():
            file_count = len(list(item.glob("*")))
            print(f"  📁 {item.name}/ ({file_count} files)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  📄 {item.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    cleanup_rank_logs()
