#!/usr/bin/env python
"""
Complete System Startup Script - Legacy Wrapper
===============================================

This script now uses the new SystemManager implementation.
Use system_manager.py directly for more control.
"""

import sys
from system_manager import main

if __name__ == "__main__":
    print("🔄 Using new SystemManager implementation...")
    print("💡 For more options, use: python system_manager.py --help")
    sys.exit(main())