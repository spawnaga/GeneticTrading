
import smtplib
import logging
import time
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class TrainingNotificationSystem:
    """
    Email notification system for training progress monitoring
    """
    
    def __init__(self, 
                 recipient_email: str = "ali.aloraibi@outlook.com",
                 phone_number: str = "+16233133816",
                 notification_interval_hours: int = 6,
                 smtp_server: str = "smtp-mail.outlook.com",
                 smtp_port: int = 587,
                 sender_email: Optional[str] = None,
                 sender_password: Optional[str] = None):
        
        self.recipient_email = recipient_email
        self.phone_number = phone_number
        self.notification_interval = notification_interval_hours * 3600  # Convert to seconds
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        
        # Use environment variables for email credentials
        self.sender_email = sender_email or os.getenv('NOTIFICATION_EMAIL')
        self.sender_password = sender_password or os.getenv('NOTIFICATION_PASSWORD')
        
        self.last_notification_time = 0
        self.training_start_time = time.time()
        self.training_metrics = []
        self.is_running = False
        self.monitoring_thread = None
        
        # Thresholds for stopping training
        self.min_improvement_threshold = 0.001  # 0.1% minimum improvement
        self.stagnation_epochs = 50  # Stop if no improvement for 50 epochs
        self.max_loss_threshold = 1000  # Stop if loss exceeds this
        
        logger.info(f"Email notifications configured for {recipient_email}")
        logger.info(f"Notifications will be sent every {notification_interval_hours} hours")
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Training monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Training monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            current_time = time.time()
            if current_time - self.last_notification_time >= self.notification_interval:
                try:
                    self.send_progress_update()
                    self.last_notification_time = current_time
                except Exception as e:
                    logger.error(f"Failed to send progress update: {e}")
            
            time.sleep(300)  # Check every 5 minutes
    
    def log_training_metrics(self, epoch: int, policy_loss: float, value_loss: float, 
                           entropy: float, avg_reward: float = 0.0, additional_metrics: Dict = None):
        """Log training metrics for analysis"""
        metrics = {
            'timestamp': time.time(),
            'epoch': epoch,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'avg_reward': avg_reward,
            'additional_metrics': additional_metrics or {}
        }
        
        self.training_metrics.append(metrics)
        
        # Keep only last 1000 metrics to prevent memory issues
        if len(self.training_metrics) > 1000:
            self.training_metrics = self.training_metrics[-1000:]
        
        # Check if training should be stopped
        if self._should_stop_training():
            self.send_stop_training_alert()
            return True
        
        return False
    
    def _should_stop_training(self) -> bool:
        """Determine if training should be stopped based on metrics"""
        if len(self.training_metrics) < self.stagnation_epochs:
            return False
        
        recent_metrics = self.training_metrics[-self.stagnation_epochs:]
        
        # Check for loss explosion
        latest_policy_loss = recent_metrics[-1]['policy_loss']
        latest_value_loss = recent_metrics[-1]['value_loss']
        
        if abs(latest_policy_loss) > self.max_loss_threshold or abs(latest_value_loss) > self.max_loss_threshold:
            logger.warning(f"Training stopped: Loss explosion detected (policy: {latest_policy_loss}, value: {latest_value_loss})")
            return True
        
        # Check for stagnation
        if len(recent_metrics) >= self.stagnation_epochs:
            early_avg_loss = np.mean([m['value_loss'] for m in recent_metrics[:10]])
            recent_avg_loss = np.mean([m['value_loss'] for m in recent_metrics[-10:]])
            
            improvement = (early_avg_loss - recent_avg_loss) / early_avg_loss
            
            if improvement < self.min_improvement_threshold:
                logger.warning(f"Training stopped: Insufficient improvement ({improvement:.4f}) over {self.stagnation_epochs} epochs")
                return True
        
        return False
    
    def _calculate_progress_stats(self) -> Dict:
        """Calculate training progress statistics"""
        if not self.training_metrics:
            return {}
        
        recent_metrics = self.training_metrics[-100:] if len(self.training_metrics) >= 100 else self.training_metrics
        latest_metrics = self.training_metrics[-1]
        
        # Calculate averages
        avg_policy_loss = np.mean([m['policy_loss'] for m in recent_metrics])
        avg_value_loss = np.mean([m['value_loss'] for m in recent_metrics])
        avg_entropy = np.mean([m['entropy'] for m in recent_metrics])
        avg_reward = np.mean([m['avg_reward'] for m in recent_metrics])
        
        # Calculate trends
        if len(recent_metrics) >= 20:
            early_loss = np.mean([m['value_loss'] for m in recent_metrics[:10]])
            recent_loss = np.mean([m['value_loss'] for m in recent_metrics[-10:]])
            loss_trend = "Improving" if recent_loss < early_loss else "Worsening"
            improvement_rate = ((early_loss - recent_loss) / early_loss) * 100
        else:
            loss_trend = "Insufficient data"
            improvement_rate = 0
        
        runtime_hours = (time.time() - self.training_start_time) / 3600
        
        return {
            'current_epoch': latest_metrics['epoch'],
            'total_epochs_completed': len(self.training_metrics),
            'runtime_hours': runtime_hours,
            'latest_policy_loss': latest_metrics['policy_loss'],
            'latest_value_loss': latest_metrics['value_loss'],
            'latest_entropy': latest_metrics['entropy'],
            'avg_policy_loss': avg_policy_loss,
            'avg_value_loss': avg_value_loss,
            'avg_entropy': avg_entropy,
            'avg_reward': avg_reward,
            'loss_trend': loss_trend,
            'improvement_rate': improvement_rate
        }
    
    def send_progress_update(self):
        """Send a progress update email"""
        try:
            stats = self._calculate_progress_stats()
            
            if not stats:
                logger.info("No training metrics available for progress update")
                return
            
            subject = f"Training Progress Update - Epoch {stats['current_epoch']}"
            
            # Create email body
            body = f"""
Training Progress Report
========================

Training Status: RUNNING
Current Epoch: {stats['current_epoch']}
Total Epochs Completed: {stats['total_epochs_completed']}
Runtime: {stats['runtime_hours']:.2f} hours

Latest Metrics:
- Policy Loss: {stats['latest_policy_loss']:.4f}
- Value Loss: {stats['latest_value_loss']:.4f}
- Entropy: {stats['latest_entropy']:.4f}

Recent Averages (last 100 epochs):
- Avg Policy Loss: {stats['avg_policy_loss']:.4f}
- Avg Value Loss: {stats['avg_value_loss']:.4f}
- Avg Entropy: {stats['avg_entropy']:.4f}
- Avg Reward: {stats['avg_reward']:.4f}

Performance Analysis:
- Loss Trend: {stats['loss_trend']}
- Improvement Rate: {stats['improvement_rate']:.2f}%

Training appears to be {"productive" if stats['improvement_rate'] > 0 else "struggling"}

Next update in {self.notification_interval // 3600} hours.

---
Automated Training Monitor
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            self._send_email(subject, body)
            logger.info(f"Progress update sent to {self.recipient_email}")
            
        except Exception as e:
            logger.error(f"Failed to send progress update: {e}")
    
    def send_stop_training_alert(self):
        """Send alert when training should be stopped"""
        try:
            subject = "🚨 TRAINING ALERT: Stopping Recommended"
            
            stats = self._calculate_progress_stats()
            
            body = f"""
TRAINING STOP ALERT
==================

⚠️ The training monitoring system recommends stopping the current training session.

Reason: Training appears to have stagnated or is showing signs of instability.

Current Status:
- Epoch: {stats.get('current_epoch', 'Unknown')}
- Runtime: {stats.get('runtime_hours', 0):.2f} hours
- Latest Value Loss: {stats.get('latest_value_loss', 'Unknown')}
- Recent Improvement: {stats.get('improvement_rate', 0):.4f}%

Recommendation: Stop training and review hyperparameters or data quality.

---
Automated Training Monitor
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            self._send_email(subject, body)
            logger.warning(f"Stop training alert sent to {self.recipient_email}")
            
        except Exception as e:
            logger.error(f"Failed to send stop training alert: {e}")
    
    def send_training_complete(self, final_stats: Dict = None):
        """Send notification when training completes"""
        try:
            subject = "✅ Training Completed Successfully"
            
            stats = final_stats or self._calculate_progress_stats()
            
            body = f"""
TRAINING COMPLETED
==================

🎉 Your trading model training has completed successfully!

Final Results:
- Total Epochs: {stats.get('total_epochs_completed', 'Unknown')}
- Total Runtime: {stats.get('runtime_hours', 0):.2f} hours
- Final Policy Loss: {stats.get('latest_policy_loss', 'Unknown')}
- Final Value Loss: {stats.get('latest_value_loss', 'Unknown')}
- Final Entropy: {stats.get('latest_entropy', 'Unknown')}

Performance Summary:
- Overall Improvement: {stats.get('improvement_rate', 0):.2f}%
- Training Status: {"Successful" if stats.get('improvement_rate', 0) > 0 else "Completed with concerns"}

Your trained model is ready for evaluation and deployment.

---
Automated Training Monitor
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            self._send_email(subject, body)
            logger.info(f"Training completion notification sent to {self.recipient_email}")
            
        except Exception as e:
            logger.error(f"Failed to send completion notification: {e}")
    
    def send_error_alert(self, error_message: str, traceback_info: str = None):
        """Send alert when training encounters an error"""
        try:
            subject = "🚨 TRAINING ERROR ALERT"
            
            body = f"""
TRAINING ERROR ALERT
===================

❌ Your training session has encountered an error and may have stopped.

Error Details:
{error_message}

{"Traceback:" if traceback_info else ""}
{traceback_info or "No additional traceback information available."}

Please check your training logs and restart if necessary.

---
Automated Training Monitor
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            self._send_email(subject, body)
            logger.error(f"Error alert sent to {self.recipient_email}")
            
        except Exception as e:
            logger.error(f"Failed to send error alert: {e}")
    
    def _send_email(self, subject: str, body: str):
        """Send email using SMTP"""
        if not self.sender_email or not self.sender_password:
            logger.warning("Email credentials not configured. Set NOTIFICATION_EMAIL and NOTIFICATION_PASSWORD environment variables.")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {self.recipient_email}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise

# Global notification system instance
notification_system = None

def initialize_notifications(email: str = "ali.aloraibi@outlook.com", 
                           phone: str = "+16233133816",
                           interval_hours: int = 6):
    """Initialize the global notification system"""
    global notification_system
    notification_system = TrainingNotificationSystem(
        recipient_email=email,
        phone_number=phone,
        notification_interval_hours=interval_hours
    )
    notification_system.start_monitoring()
    return notification_system

def log_training_progress(epoch: int, policy_loss: float, value_loss: float, 
                         entropy: float, avg_reward: float = 0.0, 
                         additional_metrics: Dict = None) -> bool:
    """Log training progress and check if training should stop"""
    global notification_system
    if notification_system:
        return notification_system.log_training_metrics(
            epoch, policy_loss, value_loss, entropy, avg_reward, additional_metrics
        )
    return False

def send_completion_notification(final_stats: Dict = None):
    """Send training completion notification"""
    global notification_system
    if notification_system:
        notification_system.send_training_complete(final_stats)
        notification_system.stop_monitoring()

def send_error_notification(error_message: str, traceback_info: str = None):
    """Send error notification"""
    global notification_system
    if notification_system:
        notification_system.send_error_alert(error_message, traceback_info)
