
#!/usr/bin/env python
"""
Training Runner with Email Notifications
=======================================

Runs the trading system with email progress notifications.
"""

import subprocess
import sys
import time
import logging
from pathlib import Path
from email_notifications import TrainingNotificationManager

def setup_logging():
    """Setup logging for the runner."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('./logs/notification_runner.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🤖 Trading AI with Email Notifications")
    print("=" * 50)
    
    # Setup email notifications
    email_manager = TrainingNotificationManager()
    
    if not email_manager.setup_from_config():
        print("⚠️  Email notifications not configured.")
        print("Run 'python setup_email_notifications.py' to configure email alerts.")
        choice = input("Continue without email notifications? (y/n): ").lower().strip()
        if choice != 'y':
            print("Setup email notifications first, then run this script again.")
            return
        email_manager = None
    else:
        print("✅ Email notifications configured")
        if email_manager.start_notifications():
            print("📧 Email monitoring started - reports every 6 hours")
        
        # Send startup notification
        test_choice = input("Send startup notification email? (y/n): ").lower().strip()
        if test_choice == 'y':
            if email_manager.email_service:
                subject = "🚀 Trading AI Training Started"
                body = f"""
                <html>
                <body>
                <h2>🚀 Trading AI Training Session Started</h2>
                <p><strong>Start Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Your trading AI training has begun. You will receive progress reports every 6 hours.</p>
                <p>Monitor the training progress and receive alerts for any issues.</p>
                </body>
                </html>
                """
                email_manager.email_service.send_email(subject, body, is_html=True)
                print("📧 Startup notification sent")
    
    # Get training arguments
    print("\n🎯 Training Configuration")
    data_percentage = input("Data percentage (0.1 for 10%, 1.0 for 100%): ").strip() or "0.1"
    max_rows = input("Max rows (0 for all): ").strip() or "5000"
    iterations = input("Adaptive iterations (default 20): ").strip() or "20"
    
    # Start training
    cmd = [
        sys.executable, "main.py",
        "--data-percentage", data_percentage,
        "--max-rows", max_rows,
        "--adaptive-iterations", iterations,
        "--log-level", "INFO"
    ]
    
    print(f"\n🚀 Starting training with command: {' '.join(cmd)}")
    print("📊 Monitor progress in ./logs/ directory")
    if email_manager:
        print("📧 Email reports will be sent every 6 hours")
    print("Press Ctrl+C to stop training\n")
    
    try:
        # Run training
        process = subprocess.Popen(cmd)
        process.wait()
        
        # Send completion notification
        if email_manager and email_manager.email_service:
            subject = "✅ Trading AI Training Completed"
            body = f"""
            <html>
            <body>
            <h2>✅ Trading AI Training Session Completed</h2>
            <p><strong>End Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Your trading AI training session has finished.</p>
            <p>Check the logs and model files for results.</p>
            </body>
            </html>
            """
            email_manager.email_service.send_email(subject, body, is_html=True)
            
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        if email_manager and email_manager.email_service:
            subject = "🛑 Trading AI Training Interrupted"
            body = f"""
            <html>
            <body>
            <h2>🛑 Training Session Interrupted</h2>
            <p><strong>Stop Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Training was stopped by user intervention.</p>
            </body>
            </html>
            """
            email_manager.email_service.send_email(subject, body, is_html=True)
    
    finally:
        if email_manager:
            email_manager.stop_notifications()
            print("📧 Email monitoring stopped")

if __name__ == "__main__":
    main()
