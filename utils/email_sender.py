import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

def send_logs_email():
    """Send all log files via email."""
    try:
        # Get email configuration from environment variables
        sender_email = os.getenv('EMAIL_SENDER')
        sender_password = os.getenv('EMAIL_PASSWORD')
        recipient_email = os.getenv('EMAIL_RECIPIENT')
        smtp_server = os.getenv('SMTP_SERVER')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        app_name = os.getenv('APP_NAME', 'Longfed Client')  # Default to 'SwaFL Client' if not set

        if not all([sender_email, sender_password, recipient_email, smtp_server]):
            logger.error("Missing email configuration in environment variables")
            return False

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f'{app_name} Training Logs - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        # Add body with app name
        body = f"""
This email contains training logs from {app_name}.

Training session has completed successfully.
Please find attached the detailed log files from the training session.

Application: {app_name}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
        msg.attach(MIMEText(body, 'plain'))

        # Attach all files from the logs directory
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        print(logs_dir)
        if os.path.exists(logs_dir):
            for filename in os.listdir(logs_dir):
                filepath = os.path.join(logs_dir, filename)
                with open(filepath, 'rb') as f:
                    attachment = MIMEApplication(f.read(), _subtype='csv')
                    attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                    msg.attach(attachment)

        # Send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        logger.info(f"Log files sent successfully via email for {app_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        return False 