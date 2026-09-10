#!/usr/bin/env python3
"""
Email Notifier with PDF Report Attachment
Updated version of run_pipeline_background_v2.py to include PDF reports
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class EmailNotifierWithReport:
    """Send emails with PDF report attachments"""
    
    def __init__(self, sender_email, sender_password, smtp_server="smtp.gmail.com", 
                 smtp_port=587, use_tls=True):
        """
        Initialize email sender
        
        Args:
            sender_email: Email address to send from
            sender_password: Password/API key
            smtp_server: SMTP server
            smtp_port: SMTP port
            use_tls: Use TLS encryption
        """
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.use_tls = use_tls
    
    def send_success_with_report(self, recipient_email, pipeline_args, pdf_path=None):
        """Send success email with optional PDF report"""
        subject = "✓ Pipeline Completed Successfully"
        
        body = f"""
Pipeline Completed Successfully!

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Arguments:
{json.dumps(pipeline_args, indent=2)}

The detailed report is attached as a PDF.

Log File: pipeline_background.log
Results are available in the outputs directory.
"""
        
        self._send_email_with_attachment(
            recipient_email, 
            subject, 
            body,
            attachment_path=pdf_path
        )
    
    def send_failure_with_report(self, recipient_email, pipeline_args, error_message, 
                                 pdf_path=None, log_tail=None):
        """Send failure email with optional PDF report and logs"""
        subject = "✗ Pipeline Failed"
        
        body = f"""
Pipeline Failed!

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Arguments:
{json.dumps(pipeline_args, indent=2)}

Error:
{error_message}
"""
        
        if log_tail:
            body += f"\n\nLast 20 lines of log:\n{log_tail}"
        
        body += "\n\nLog File: pipeline_background.log"
        
        self._send_email_with_attachment(
            recipient_email,
            subject,
            body,
            attachment_path=pdf_path
        )
    
    def _send_email_with_attachment(self, recipient_email, subject, body, 
                                   attachment_path=None):
        """Send email with optional file attachment"""
        try:
            logger.info(f"Sending email to {recipient_email}...")
            
            # Create message
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = recipient_email
            message["Subject"] = subject
            
            # Add body
            message.attach(MIMEText(body, "plain"))
            
            # Add attachment if provided
            if attachment_path:
                self._attach_file(message, attachment_path)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
            
            logger.info(f"✓ Email sent to {recipient_email}")
            if attachment_path:
                logger.info(f"✓ Attachment included: {attachment_path}")
        
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"✗ Authentication failed: {e}")
            logger.error("Check your email and password/app password")
            raise
        except Exception as e:
            logger.error(f"✗ Failed to send email: {e}")
            raise
    
    def _attach_file(self, message, file_path):
        """Attach a file to an email message"""
        try:
            from pathlib import Path
            
            path = Path(file_path)
            
            if not path.exists():
                logger.warning(f"Attachment file not found: {file_path}")
                return
            
            # Determine MIME type
            if path.suffix.lower() == '.pdf':
                mime_type = 'application/pdf'
                subtype = 'pdf'
            elif path.suffix.lower() in ['.txt', '.log']:
                mime_type = 'text/plain'
                subtype = 'plain'
            else:
                mime_type = 'application/octet-stream'
                subtype = 'octet-stream'
            
            # Create attachment
            with open(file_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            # Encode and attach
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {path.name}'
            )
            message.attach(part)
            
            logger.info(f"✓ File attached: {path.name}")
        
        except Exception as e:
            logger.error(f"Error attaching file: {e}")
            # Don't fail the entire email send if attachment fails
            pass


# Example usage in pipeline runner:
def example_usage():
    """
    Example of how to use this in run_pipeline_background.py
    """
    
    from pipeline_report_generator import generate_pdf_report
    
    # After pipeline completes:
    try:
        # Generate PDF report
        pipeline_results = {
            'pipeline_id': 'test-123',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {...},
            'training_results': {...},
            'feature_importance': {...}
        }
        
        pdf_path = generate_pdf_report(pipeline_results, "pipeline_report.pdf")
        
        # Send email with PDF
        notifier = EmailNotifierWithReport(
            sender_email="your.email@gmail.com",
            sender_password="your_app_password"
        )
        
        notifier.send_success_with_report(
            recipient_email="user@example.com",
            pipeline_args={...},
            pdf_path=pdf_path  # Attach the PDF!
        )
    
    except Exception as e:
        # Send failure email with whatever we have
        notifier.send_failure_with_report(
            recipient_email="user@example.com",
            pipeline_args={...},
            error_message=str(e),
            pdf_path=None  # PDF might not exist if it failed early
        )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    print("Email notifier with PDF attachment support loaded")
    print("See example_usage() for integration example")
