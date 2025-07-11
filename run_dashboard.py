
#!/usr/bin/env python
"""
Run Complete Dashboard System
============================

Simple script to start the comprehensive trading dashboard
with proper data formatting and TensorBoard integration.
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the complete dashboard system."""
    logger.info("🚀 Starting Revolutionary NQ Futures Trading Dashboard")
    
    # Step 1: Fix data formatting
    logger.info("📊 Step 1: Checking and fixing data format...")
    try:
        import fix_nq_format
        fix_nq_format.fix_nq_data_format()
        logger.info("✅ Data formatting complete")
    except Exception as e:
        logger.warning(f"Data formatting had issues: {e}")
        logger.info("Creating sample data...")
        fix_nq_format.create_sample_nq_data()
    
    # Step 2: Start the comprehensive dashboard
    logger.info("🌐 Step 2: Starting comprehensive dashboard...")
    try:
        from start_dashboard import main as start_main
        start_main()
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        
        # Fallback: start simple dashboard
        logger.info("🔄 Starting fallback dashboard...")
        from standalone_dashboard import main as dashboard_main
        dashboard_main()

if __name__ == "__main__":
    main()
