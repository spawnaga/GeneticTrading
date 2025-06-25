
#!/usr/bin/env python
"""
Email Notification Service for Training Progress
==============================================

Sends periodic email updates about training progress and recommendations.
"""

import smtplib
import json
import time
import threading
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class EmailNotificationService:
    """Service for sending training progress notifications via email."""
    
    def __init__(self, 
                 smtp_server: str = "smtp.gmail.com",
                 smtp_port: int = 587,
                 sender_email: str = "",
                 sender_password: str = "",
                 recipient_emails: List[str] = None,
                 log_dir: str = "./logs"):
        
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_emails = recipient_emails or []
        self.log_dir = Path(log_dir)
        
        # Notification settings
        self.notification_interval = 6 * 3600  # 6 hours in seconds
        self.last_notification = None
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Training state tracking
        self.last_check_time = datetime.now()
        self.performance_threshold = -0.1  # Stop if performance drops below this
        self.stagnation_hours = 12  # Stop if no improvement for 12 hours
        
    def send_email(self, subject: str, body: str, is_html: bool = False):
        """Send an email notification."""
        if not self.sender_email or not self.sender_password or not self.recipient_emails:
            logger.warning("Email credentials not configured. Skipping email notification.")
            return False
            
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MimeText(body, 'html' if is_html else 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, self.recipient_emails, text)
            server.quit()
            
            logger.info(f"Email notification sent successfully to {self.recipient_emails}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def generate_progress_email(self, training_data: Dict) -> tuple:
        """Generate email subject and body from training data."""
        if not training_data or 'performance_history' not in training_data:
            subject = "🤖 Trading Training - No Data Available"
            body = """
            <h2>Training Progress Report</h2>
            <p><strong>Status:</strong> No training data available</p>
            <p>The training system may not be running or no metrics have been logged yet.</p>
            """
            return subject, body
            
        history = training_data['performance_history']
        warnings = training_data.get('warning_flags', [])
        
        if not history:
            subject = "🤖 Trading Training - Starting Up"
            body = """
            <h2>Training Progress Report</h2>
            <p><strong>Status:</strong> Training initialization in progress</p>
            """
            return subject, body
        
        # Get latest metrics
        latest = history[-1]
        total_iterations = len(history)
        current_performance = latest.get('performance', 0)
        current_method = latest.get('method', 'Unknown')
        metrics = latest.get('metrics', {})
        
        # Calculate progress indicators
        best_performance = max([h.get('performance', 0) for h in history])
        total_time = sum([h.get('training_time', 0) for h in history]) / 3600  # Convert to hours
        avg_time_per_iteration = (total_time / total_iterations * 60) if total_iterations > 0 else 0  # Minutes
        
        # Determine status and recommendation
        recommendation = self._get_recommendation(training_data)
        status_emoji = self._get_status_emoji(recommendation)
        
        # Recent trend
        if len(history) >= 3:
            recent_performances = [h.get('performance', 0) for h in history[-3:]]
            trend = "📈 Improving" if recent_performances[-1] > recent_performances[0] else "📉 Declining"
        else:
            trend = "📊 Establishing baseline"
        
        # Warning summary
        warning_summary = ""
        if warnings:
            recent_warnings = warnings[-3:]  # Last 3 warnings
            warning_summary = f"""
            <h3>⚠️ Recent Warnings ({len(warnings)} total)</h3>
            <ul>
            {''.join([f"<li>{w}</li>" for w in recent_warnings])}
            </ul>
            """
        
        # Create subject
        subject = f"{status_emoji} Trading Training - Iter {total_iterations} | Perf: {current_performance:.4f}"
        
        # Create HTML body
        body = f"""
        <html>
        <body>
        <h2>🤖 Trading AI Training Progress Report</h2>
        <p><strong>Report Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h3>📊 Current Status</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr><td><strong>Iterations Completed</strong></td><td>{total_iterations}</td></tr>
            <tr><td><strong>Current Method</strong></td><td>{current_method}</td></tr>
            <tr><td><strong>Current Performance</strong></td><td>{current_performance:.4f}</td></tr>
            <tr><td><strong>Best Performance</strong></td><td>{best_performance:.4f}</td></tr>
            <tr><td><strong>Performance Efficiency</strong></td><td>{(current_performance/best_performance*100) if best_performance > 0 else 0:.1f}%</td></tr>
            <tr><td><strong>Recent Trend</strong></td><td>{trend}</td></tr>
        </table>
        
        <h3>🎯 Trading Metrics</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr><td><strong>CAGR</strong></td><td>{metrics.get('cagr', 0):.2f}%</td></tr>
            <tr><td><strong>Sharpe Ratio</strong></td><td>{metrics.get('sharpe', 0):.2f}</td></tr>
            <tr><td><strong>Max Drawdown</strong></td><td>{metrics.get('mdd', 0):.2f}%</td></tr>
            <tr><td><strong>Total Profit</strong></td><td>${metrics.get('total_profit', 0):,.2f}</td></tr>
            <tr><td><strong>Win Rate</strong></td><td>{metrics.get('win_rate', 0)*100:.1f}%</td></tr>
        </table>
        
        <h3>⏱️ Time Statistics</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr><td><strong>Total Training Time</strong></td><td>{total_time:.1f} hours</td></tr>
            <tr><td><strong>Avg Time/Iteration</strong></td><td>{avg_time_per_iteration:.1f} minutes</td></tr>
        </table>
        
        {warning_summary}
        
        <h3>🚨 Recommendation</h3>
        <p><strong>Status:</strong> {recommendation}</p>
        {self._get_recommendation_details(recommendation)}
        
        <h3>📈 Next Steps</h3>
        {self._get_next_steps_html(recommendation)}
        
        <hr>
        <p><em>This is an automated report from your Trading AI system. Next report in 6 hours.</em></p>
        </body>
        </html>
        """
        
        return subject, body
    
    def _get_recommendation(self, training_data: Dict) -> str:
        """Get training recommendation based on data."""
        warnings = training_data.get('warning_flags', [])
        history = training_data.get('performance_history', [])
        
        if not history:
            return "STARTING"
        
        # Count severe warnings
        severe_warnings = len([w for w in warnings if any(keyword in w for keyword in 
                              ["STAGNATION", "PERFORMANCE_COLLAPSE", "EXCESSIVE_SWITCHING"])])
        
        if severe_warnings >= 3:
            return "STOP_IMMEDIATELY"
        elif severe_warnings >= 2:
            return "CONSIDER_STOPPING"
        elif len(warnings) >= 5:
            return "REVIEW_HYPERPARAMETERS"
        else:
            return "CONTINUE"
    
    def _get_status_emoji(self, recommendation: str) -> str:
        """Get status emoji based on recommendation."""
        emoji_map = {
            "CONTINUE": "✅",
            "REVIEW_HYPERPARAMETERS": "⚠️",
            "CONSIDER_STOPPING": "🚨",
            "STOP_IMMEDIATELY": "🛑",
            "STARTING": "🚀"
        }
        return emoji_map.get(recommendation, "📊")
    
    def _get_recommendation_details(self, recommendation: str) -> str:
        """Get detailed recommendation explanation."""
        details = {
            "CONTINUE": "<p style='color: green;'>Training is progressing well. Continue monitoring.</p>",
            "REVIEW_HYPERPARAMETERS": "<p style='color: orange;'>Multiple warnings detected. Consider reviewing learning rates, batch sizes, and switching thresholds.</p>",
            "CONSIDER_STOPPING": "<p style='color: red;'>Serious issues detected. Training may not be productive. Consider stopping if no improvement in next few iterations.</p>",
            "STOP_IMMEDIATELY": "<p style='color: red; font-weight: bold;'>Critical issues detected. Recommend stopping training immediately to save compute resources.</p>",
            "STARTING": "<p style='color: blue;'>Training initialization in progress.</p>"
        }
        return details.get(recommendation, "")
    
    def _get_next_steps_html(self, recommendation: str) -> str:
        """Get next steps recommendations in HTML format."""
        steps = {
            "CONTINUE": "<ul><li>Continue training - progress looks good</li><li>Monitor for sustained improvement</li></ul>",
            "REVIEW_HYPERPARAMETERS": "<ul><li>Review learning rates, batch sizes, and other hyperparameters</li><li>Consider adjusting adaptive switching thresholds</li></ul>",
            "CONSIDER_STOPPING": "<ul><li>Consider stopping if performance doesn't improve in next 2-3 iterations</li><li>Monitor closely for improvements</li></ul>",
            "STOP_IMMEDIATELY": "<ul><li>Stop training immediately to save compute resources</li><li>Review hyperparameters and training setup</li></ul>",
            "STARTING": "<ul><li>Monitor initial progress</li><li>Ensure data preprocessing completed successfully</li></ul>"
        }
        return steps.get(recommendation, "<ul><li>Continue monitoring</li></ul>")
    
    def check_and_send_notification(self):
        """Check if notification should be sent and send it."""
        try:
            # Load training metrics
            metrics_file = self.log_dir / "training_metrics.json"
            
            if not metrics_file.exists():
                logger.info("No training metrics file found. Training may not have started yet.")
                return
            
            with open(metrics_file, 'r') as f:
                training_data = json.load(f)
            
            # Check if we should send notification
            now = datetime.now()
            if (self.last_notification is None or 
                (now - self.last_notification).total_seconds() >= self.notification_interval):
                
                # Generate and send email
                subject, body = self.generate_progress_email(training_data)
                
                if self.send_email(subject, body, is_html=True):
                    self.last_notification = now
                    logger.info("Periodic training notification sent")
                
                # Check if we should recommend stopping
                recommendation = self._get_recommendation(training_data)
                if recommendation in ["STOP_IMMEDIATELY", "CONSIDER_STOPPING"]:
                    self.send_urgent_notification(recommendation, training_data)
                    
        except Exception as e:
            logger.error(f"Error in notification check: {e}")
    
    def send_urgent_notification(self, recommendation: str, training_data: Dict):
        """Send urgent notification for critical issues."""
        subject = f"🚨 URGENT: Trading Training - {recommendation}"
        
        history = training_data.get('performance_history', [])
        warnings = training_data.get('warning_flags', [])
        
        body = f"""
        <html>
        <body>
        <h2>🚨 URGENT: Training Issue Detected</h2>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Recommendation:</strong> {recommendation}</p>
        
        <h3>⚠️ Warning Summary</h3>
        <ul>
        {''.join([f"<li>{w}</li>" for w in warnings[-5:]])}  
        </ul>
        
        <h3>📊 Current Status</h3>
        <p><strong>Iterations:</strong> {len(history)}</p>
        {f"<p><strong>Latest Performance:</strong> {history[-1].get('performance', 0):.4f}</p>" if history else ""}
        
        <h3>🚨 Action Required</h3>
        {self._get_recommendation_details(recommendation)}
        
        <p><em>This is an urgent automated alert. Please review your training setup.</em></p>
        </body>
        </html>
        """
        
        self.send_email(subject, body, is_html=True)
        logger.warning(f"Urgent notification sent: {recommendation}")
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Email monitoring started - notifications every 6 hours")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Email monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in background thread."""
        while self.monitoring_active:
            try:
                self.check_and_send_notification()
                # Check every 10 minutes for urgent issues, every 6 hours for regular updates
                time.sleep(600)  # 10 minutes
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(300)  # 5 minutes on error

# Integration class for easy setup
class TrainingNotificationManager:
    """Manages training notifications with easy configuration."""
    
    def __init__(self):
        self.email_service = None
        self.config_file = Path("./config/email_config.json")
        
    def setup_from_config(self, config_path: str = None):
        """Setup email notifications from config file."""
        config_path = Path(config_path) if config_path else self.config_file
        
        if not config_path.exists():
            self.create_config_template(config_path)
            logger.info(f"Created email config template at {config_path}")
            logger.info("Please fill in your email credentials and run setup again")
            return False
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.email_service = EmailNotificationService(
                smtp_server=config.get('smtp_server', 'smtp.gmail.com'),
                smtp_port=config.get('smtp_port', 587),
                sender_email=config.get('sender_email', ''),
                sender_password=config.get('sender_password', ''),
                recipient_emails=config.get('recipient_emails', [])
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup email notifications: {e}")
            return False
    
    def create_config_template(self, config_path: Path):
        """Create email configuration template."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        template = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "your-email@gmail.com",
            "sender_password": "your-app-password",
            "recipient_emails": ["your-email@gmail.com"],
            "notification_interval_hours": 6,
            "instructions": {
                "gmail": "Use Gmail App Password, not regular password",
                "smtp_server": "Common servers: smtp.gmail.com, smtp.outlook.com, smtp.yahoo.com",
                "recipient_emails": "List of emails to receive notifications"
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(template, f, indent=2)
    
    def start_notifications(self):
        """Start email notifications."""
        if not self.email_service:
            logger.error("Email service not configured. Run setup_from_config() first.")
            return False
            
        self.email_service.start_monitoring()
        return True
    
    def stop_notifications(self):
        """Stop email notifications."""
        if self.email_service:
            self.email_service.stop_monitoring()
    
    def send_test_email(self):
        """Send a test email to verify configuration."""
        if not self.email_service:
            logger.error("Email service not configured")
            return False
            
        subject = "🤖 Trading AI - Test Notification"
        body = """
        <h2>Test Email from Trading AI System</h2>
        <p>This is a test email to verify your notification setup is working correctly.</p>
        <p><strong>Time:</strong> {}</p>
        <p>If you receive this email, your notification system is configured properly!</p>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        return self.email_service.send_email(subject, body, is_html=True)
