#!/usr/bin/env python3
"""
Background Pipeline Runner with Email Notification
Supports: Gmail, Outlook, Yahoo, SMTP, Environment Variables

Usage:
    # Using environment variables (most secure)
    export EMAIL_USER=your.email@gmail.com
    export EMAIL_PASSWORD=your_password
    python run_pipeline_background_v2.py \
        --tissue liver \
        -tc "Factor Value[Spaceflight]" \
        --email recipient@example.com
    
    # Using different provider
    python run_pipeline_background_v2.py \
        --tissue liver \
        -tc "Factor Value[Spaceflight]" \
        --email recipient@example.com \
        --provider outlook
    
    # Using custom SMTP server
    python run_pipeline_background_v2.py \
        --tissue liver \
        -tc "Factor Value[Spaceflight]" \
        --email recipient@example.com \
        --smtp-server smtp.yourcompany.com \
        --smtp-port 587
"""

import subprocess
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
import getpass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_background.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Email provider configurations
EMAIL_PROVIDERS = {
    'gmail': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'use_tls': True,
        'note': 'Gmail: Requires app password (https://myaccount.google.com/apppasswords)'
    },
    'outlook': {
        'smtp_server': 'smtp.office365.com',
        'smtp_port': 587,
        'use_tls': True,
        'note': 'Outlook: Use your Outlook password'
    },
    'yahoo': {
        'smtp_server': 'smtp.mail.yahoo.com',
        'smtp_port': 587,
        'use_tls': True,
        'note': 'Yahoo: Requires app password (https://login.yahoo.com/account/security)'
    },
    'sendgrid': {
        'smtp_server': 'smtp.sendgrid.net',
        'smtp_port': 587,
        'use_tls': True,
        'note': 'SendGrid: Username is "apikey", Password is your API key'
    },
    'mailgun': {
        'smtp_server': 'smtp.mailgun.org',
        'smtp_port': 587,
        'use_tls': True,
        'note': 'Mailgun: Use your SMTP credentials'
    },
}


class EmailNotifier:
    """Send email notifications via SMTP"""
    
    def __init__(self, sender_email, sender_password, smtp_server="smtp.gmail.com", 
                 smtp_port=587, use_tls=True):
        """
        Initialize email sender
        
        Args:
            sender_email: Email address to send from
            sender_password: Password/API key for that email
            smtp_server: SMTP server address
            smtp_port: SMTP port
            use_tls: Whether to use TLS encryption
        """
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.use_tls = use_tls
    
    def send_success(self, recipient_email, pipeline_args, output_file=None):
        """Send success email"""
        subject = "✓ Pipeline Completed Successfully"
        
        body = f"""
Pipeline Completed Successfully!

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Arguments:
{json.dumps(pipeline_args, indent=2)}

Log File: pipeline_background.log

Results are available in the outputs directory.
"""
        
        if output_file:
            body += f"\n\nOutput File: {output_file}"
        
        self._send_email(recipient_email, subject, body)
    
    def send_failure(self, recipient_email, pipeline_args, error_message, log_tail=None):
        """Send failure email"""
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
        
        self._send_email(recipient_email, subject, body)
    
    def _send_email(self, recipient_email, subject, body):
        """Send email via SMTP"""
        try:
            logger.info(f"Sending email to {recipient_email}...")
            logger.info(f"Using SMTP: {self.smtp_server}:{self.smtp_port}")
            
            # Create message
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = recipient_email
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))
            
            # Send email
            if self.use_tls:
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(message)
            else:
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(message)
            
            logger.info(f"✓ Email sent to {recipient_email}")
        
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"✗ Authentication failed: {e}")
            logger.error("Check your email and password/app password")
            raise
        except smtplib.SMTPException as e:
            logger.error(f"✗ SMTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Failed to send email: {e}")
            raise


class PipelineRunner:
    """Run pipeline in background with email notification"""
    
    def __init__(self, pipeline_script="new_multi_pipeline.py"):
        self.pipeline_script = pipeline_script
        self.start_time = datetime.now()
    
    def run(self, args, email_notifier=None, recipient_email=None):
        """
        Run pipeline and send email notification
        
        Args:
            args: Dict of command line arguments
            email_notifier: EmailNotifier instance
            recipient_email: Email to send notification to
        """
        logger.info("="*60)
        logger.info("PIPELINE STARTED")
        logger.info("="*60)
        logger.info(f"Arguments: {args}")
        
        try:
            # Build command
            cmd = [sys.executable, self.pipeline_script]
            
            # Add arguments
            for key, value in args.items():
                if value is None:
                    continue
                
                if key.startswith("--"):
                    cmd.append(key)
                    if value is not True:  # Boolean flags
                        cmd.append(str(value))
                elif key.startswith("-"):
                    cmd.append(key)
                    cmd.append(str(value))
            
            logger.info(f"Running: {' '.join(cmd)}")
            
            # Run pipeline
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            logger.info("Pipeline execution completed")
            logger.info(f"Return code: {result.returncode}")
            
            if result.stdout:
                logger.info(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                logger.error(f"STDERR:\n{result.stderr}")
            
            # Send notification
            if email_notifier and recipient_email:
                if result.returncode == 0:
                    email_notifier.send_success(recipient_email, args)
                else:
                    email_notifier.send_failure(
                        recipient_email, 
                        args, 
                        result.stderr or "Unknown error",
                        log_tail=self._get_log_tail()
                    )
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"✓ Pipeline completed in {elapsed:.1f} seconds")
            
            return result.returncode
        
        except Exception as e:
            logger.error(f"✗ Pipeline failed: {e}", exc_info=True)
            
            if email_notifier and recipient_email:
                email_notifier.send_failure(
                    recipient_email,
                    args,
                    str(e),
                    log_tail=self._get_log_tail()
                )
            
            return 1
    
    def _get_log_tail(self, n_lines=20):
        """Get last N lines of log file"""
        try:
            with open("pipeline_background.log", "r") as f:
                lines = f.readlines()
            return "".join(lines[-n_lines:])
        except:
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline in background with email notification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  Using Gmail (requires app password):
    python run_pipeline_background_v2.py --tissue liver -tc "Factor Value[Spaceflight]" --email results@gmail.com

  Using Outlook:
    python run_pipeline_background_v2.py --tissue liver -tc "Factor Value[Spaceflight]" --email results@example.com --provider outlook

  Using environment variables (most secure):
    export EMAIL_USER=your.email@gmail.com
    export EMAIL_PASSWORD=your_password
    python run_pipeline_background_v2.py --tissue liver -tc "Factor Value[Spaceflight]" --email results@example.com

  Using custom SMTP server:
    python run_pipeline_background_v2.py --tissue liver -tc "Factor Value[Spaceflight]" --email results@example.com --smtp-server smtp.company.com --smtp-port 587
        """
    )
    
    # Pipeline arguments
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--tissue', type=str, help='Tissue type to combine')
    dataset_group.add_argument('--osd_ids', type=str, help='Comma-separated OSD IDs')
    
    parser.add_argument('-tc', '--target_column', required=True, help='Target column name')
    parser.add_argument('-ts', '--test_size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('-al', '--algorithm', default='random_forest', help='ML algorithm')
    parser.add_argument('-mf', '--min_features', type=int, default=1000, help='Minimum features')
    parser.add_argument('--no-ensemble', action='store_true', help='Skip ensemble')
    parser.add_argument('--no-kegg', action='store_true', help='Skip KEGG analysis')
    parser.add_argument('--no_feature_importance', action='store_true', help='Skip feature importance')
    
    # Email arguments
    parser.add_argument('--email', type=str, required=True, help='Email to notify')
    
    # Email provider/auth options
    parser.add_argument('--provider', choices=list(EMAIL_PROVIDERS.keys()), 
                       default='gmail', help='Email provider (default: gmail)')
    parser.add_argument('--smtp_server', type=str, help='Custom SMTP server (overrides provider)')
    parser.add_argument('--smtp_port', type=int, help='Custom SMTP port (overrides provider)')
    parser.add_argument('--no_tls', action='store_true', help='Disable TLS encryption')
    
    parser.add_argument('--sender_email', type=str, help='Email to send from (default: prompt or EMAIL_USER env var)')
    parser.add_argument('--sender_password', type=str, help='Password/API key (default: prompt or EMAIL_PASSWORD env var)')
    
    args = parser.parse_args()
    
    # Get provider settings
    provider_config = EMAIL_PROVIDERS.get(args.provider, {})
    smtp_server = args.smtp_server or provider_config.get('smtp_server', 'smtp.gmail.com')
    smtp_port = args.smtp_port or provider_config.get('smtp_port', 587)
    use_tls = not args.no_tls and provider_config.get('use_tls', True)
    
    logger.info(f"Email provider: {args.provider}")
    if provider_config.get('note'):
        logger.info(f"Note: {provider_config['note']}")
    
    # Get email credentials
    sender_email = args.sender_email or os.getenv('EMAIL_USER')
    sender_password = args.sender_password or os.getenv('EMAIL_PASSWORD')
    
    if not sender_email:
        sender_email = input("Email to send from: ")
    if not sender_password:
        sender_password = getpass.getpass("Password/App Password/API Key: ")
    
    # Build pipeline arguments
    pipeline_args = {
        '--target_column': args.target_column,
        '--test_size': args.test_size,
        '--algorithm': args.algorithm,
        '--min_features': args.min_features,
    }
    
    if args.tissue:
        pipeline_args['--tissue'] = args.tissue
    elif args.osd_ids:
        pipeline_args['--osd_ids'] = args.osd_ids
    
    if args.no_ensemble:
        pipeline_args['--no-ensemble'] = True
    if args.no_kegg:
        pipeline_args['--no-kegg'] = True
    if args.no_feature_importance:
        pipeline_args['--no-feature-importance'] = True
    
    # Create notifier and runner
    try:
        email_notifier = EmailNotifier(
            sender_email, 
            sender_password,
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            use_tls=use_tls
        )
        runner = PipelineRunner()
        
        # Run pipeline
        exit_code = runner.run(pipeline_args, email_notifier, args.email)
        sys.exit(exit_code)
    
    except Exception as e:
        logger.error(f"✗ Failed to initialize: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
