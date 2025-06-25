#!/usr/bin/env python3
"""
Setup script for email notifications
"""
import os
import getpass

def setup_email_credentials():
    """
    Interactive setup for email notification credentials
    """
    print("=== Email Notification Setup ===")
    print("This will configure email notifications for your training progress.")
    print()

    # Get email credentials
    print("For Outlook/Hotmail accounts:")
    print("1. Use your full email address as username")
    print("2. You may need to generate an app password if 2FA is enabled")
    print("3. Go to https://account.microsoft.com/security to create app passwords")
    print()

    sender_email = input("Enter your sender email address (or press Enter for ali.aloraibi@outlook.com): ").strip()
    if not sender_email:
        sender_email = "ali.aloraibi@outlook.com"

    sender_password = getpass.getpass("Enter your email password (or app password): ")

    # Set environment variables
    os.environ['NOTIFICATION_EMAIL'] = sender_email
    os.environ['NOTIFICATION_PASSWORD'] = sender_password

    # Create a .env file for persistence
    try:
        with open('.env', 'w') as f:
            f.write(f"NOTIFICATION_EMAIL={sender_email}\n")
            f.write(f"NOTIFICATION_PASSWORD={sender_password}\n")
        print("✅ Credentials saved to .env file")
    except Exception as e:
        print(f"❌ Failed to save credentials to .env file: {e}")

    print()
    print("✅ Email notifications configured successfully!")
    print(f"📧 Notifications will be sent to: ali.aloraibi@outlook.com")
    print(f"📱 Phone: +16233133816")
    print()
    print("Your training will now send you updates every 6 hours.")
    print("You'll be notified if training stagnates or completes.")

if __name__ == "__main__":
    setup_email_credentials()