#!/usr/bin/env python3
"""
SMTP Connection Tester for Email Agent
Tests different SMTP configurations to diagnose authentication issues.
"""

import logging
import os
import smtplib
from email.mime.text import MIMEText

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Try to load .env file manually if it exists
def load_env_file():
    env_file = "../.env"
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value.strip("'\"")


load_env_file()


def test_smtp_connection():
    """Test SMTP connection with current environment variables."""

    # Get email configuration
    sender_email = os.getenv("EMAIL")
    password = os.getenv("EMAIL_PWD")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port_str = os.getenv("SMTP_PORT", "587")

    print("=== SMTP Configuration Test ===")
    print(f"SMTP Server: {smtp_server}")
    print(f"SMTP Port: {smtp_port_str}")
    print(f"Email: {sender_email}")
    print(f"Password: {'*' * len(password) if password else 'NOT SET'}")
    print()

    if not all([sender_email, password]):
        print("❌ ERROR: Missing EMAIL or EMAIL_PWD environment variables")
        return False

    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        print(f"❌ ERROR: Invalid port number: {smtp_port_str}")
        return False

    # Test different configurations
    configs_to_test = [
        {"port": 465, "use_ssl": True, "name": "Gmail SSL (Port 465)"},
        {"port": 587, "use_ssl": False, "name": "Gmail STARTTLS (Port 587)"},
        {"port": 25, "use_ssl": False, "name": "Standard SMTP (Port 25)"},
    ]

    for config in configs_to_test:
        print(f"Testing {config['name']}...")
        success = test_config(
            smtp_server, config["port"], config["use_ssl"], sender_email, password
        )
        if success:
            print(f"✅ SUCCESS: {config['name']} works!")
            print(f"   Recommended .env settings:")
            print(f"   SMTP_SERVER={smtp_server}")
            print(f"   SMTP_PORT={config['port']}")
            return True
        print()

    print("❌ All configurations failed. Please check:")
    print("1. Email provider settings")
    print("2. App-specific password (for Gmail)")
    print("3. Two-factor authentication settings")
    print("4. Account security settings")

    return False


def test_config(server, port, use_ssl, email, password):
    """Test a specific SMTP configuration."""
    smtp_server = None
    try:
        print(f"  Connecting to {server}:{port}...")

        if use_ssl:
            smtp_server = smtplib.SMTP_SSL(server, port, timeout=10)
        else:
            smtp_server = smtplib.SMTP(server, port, timeout=10)
            print("  Starting TLS...")
            smtp_server.starttls()

        print("  Attempting login...")
        smtp_server.login(email, password)

        print("  ✅ Login successful!")
        return True

    except smtplib.SMTPAuthenticationError as e:
        print(f"  ❌ Authentication failed: {e}")
        if "Application-specific" in str(e) or "app password" in str(e).lower():
            print("  💡 Hint: Use an app-specific password for Gmail")
    except smtplib.SMTPConnectError as e:
        print(f"  ❌ Connection failed: {e}")
    except smtplib.SMTPException as e:
        print(f"  ❌ SMTP error: {e}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
    finally:
        if smtp_server:
            try:
                smtp_server.quit()
            except:
                pass

    return False


def send_test_email():
    """Send a test email using the working configuration."""
    sender_email = os.getenv("EMAIL")
    receiver_email = os.getenv("EMAIL_RECEIVER") or sender_email
    password = os.getenv("EMAIL_PWD")

    if not all([sender_email, receiver_email, password]):
        print("❌ Missing email configuration for test email")
        return

    try:
        msg = MIMEText("This is a test email from your sales assistant SMTP configuration.")
        msg["Subject"] = "🧪 SMTP Test Email"
        msg["From"] = sender_email
        msg["To"] = receiver_email

        # Try the recommended configuration (Gmail SSL)
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()

        print(f"✅ Test email sent successfully to {receiver_email}")

    except Exception as e:
        print(f"❌ Failed to send test email: {e}")


def gmail_troubleshooting_guide():
    """Print Gmail-specific troubleshooting guide."""
    print(
        """
=== Gmail SMTP Troubleshooting Guide ===

1. 🔐 Enable 2-Factor Authentication:
   - Go to Google Account settings
   - Enable 2-Factor Authentication

2. 🔑 Generate App-Specific Password:
   - Go to Google Account > Security > App passwords
   - Generate password for "Mail" application
   - Use this password in EMAIL_PWD (not your regular password)

3. ⚙️ Recommended Settings:
   EMAIL=your.email@gmail.com
   EMAIL_PWD=your_16_character_app_password
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=465

4. 🚫 Common Issues:
   - Using regular password instead of app password
   - 2FA not enabled
   - "Less secure app access" disabled (deprecated)
   - Incorrect server/port combination

5. 🔍 Alternative Settings to Try:
   - Port 587 with STARTTLS (instead of 465 SSL)
   - Port 25 for some hosting providers

6. 📧 Other Email Providers:
   Outlook: smtp.office365.com:587
   Yahoo: smtp.mail.yahoo.com:587
   """
    )


if __name__ == "__main__":
    print("Starting SMTP configuration test...\n")

    # Test SMTP connection
    if test_smtp_connection():
        print("\n" + "=" * 50)
        response = input("Would you like to send a test email? (y/n): ")
        if response.lower() == "y":
            send_test_email()

    print("\n" + "=" * 50)
    gmail_troubleshooting_guide()
