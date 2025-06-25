
#!/usr/bin/env python
"""
Email Notification Setup Script
==============================

Run this script to configure email notifications for your training progress.
"""

import json
import os
from pathlib import Path
from email_notifications import TrainingNotificationManager

def main():
    print("🤖 Trading AI - Email Notification Setup")
    print("=" * 50)
    
    manager = TrainingNotificationManager()
    config_path = Path("./config/email_config.json")
    
    if config_path.exists():
        print(f"✅ Found existing config at {config_path}")
        choice = input("Do you want to reconfigure? (y/n): ").lower().strip()
        if choice != 'y':
            print("Using existing configuration...")
            if manager.setup_from_config():
                test_choice = input("Send test email? (y/n): ").lower().strip()
                if test_choice == 'y':
                    if manager.send_test_email():
                        print("✅ Test email sent successfully!")
                    else:
                        print("❌ Failed to send test email. Check your configuration.")
            return
    
    print("\n📧 Email Configuration")
    print("Note: For Gmail, use an App Password, not your regular password")
    print("Generate App Password: https://support.google.com/accounts/answer/185833")
    
    # Get email configuration
    config = {
        "smtp_server": input("SMTP Server (default: smtp.gmail.com): ").strip() or "smtp.gmail.com",
        "smtp_port": int(input("SMTP Port (default: 587): ").strip() or "587"),
        "sender_email": input("Your email address: ").strip(),
        "sender_password": input("Your email password/app password: ").strip(),
        "recipient_emails": [],
        "notification_interval_hours": 6
    }
    
    # Get recipient emails
    print("\n📨 Recipient Email Addresses")
    print("Enter email addresses to receive notifications (press Enter when done):")
    while True:
        email = input("Email address: ").strip()
        if not email:
            break
        config["recipient_emails"].append(email)
    
    if not config["recipient_emails"]:
        config["recipient_emails"] = [config["sender_email"]]
        print(f"Using sender email as recipient: {config['sender_email']}")
    
    # Save configuration
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Configuration saved to {config_path}")
    
    # Test configuration
    print("\n🧪 Testing email configuration...")
    if manager.setup_from_config():
        if manager.send_test_email():
            print("✅ Email configuration successful!")
            print("📧 Test email sent to:", ", ".join(config["recipient_emails"]))
            print("\n🚀 Email notifications are now active!")
            print("📊 You'll receive progress reports every 6 hours")
            print("🚨 Urgent alerts will be sent immediately if training issues are detected")
        else:
            print("❌ Failed to send test email.")
            print("Please check your email credentials and try again.")
    else:
        print("❌ Failed to setup email notifications.")
        print(f"Please check the configuration in {config_path}")

if __name__ == "__main__":
    main()
