#!/usr/bin/env python3
"""
Simple Trading System Launcher
==============================

Easy-to-use script for running the trading system locally.
Perfect for PyCharm or any IDE.
"""

import os
import sys
import logging
from pathlib import Path

def setup_environment():
    """Setup basic environment for local development."""
    # Create necessary directories
    dirs_to_create = [
        './models',
        './logs', 
        './cached_data',
        './runs'
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def run_quick_test():
    """Run a quick test with minimal data for development."""
    print("🚀 Starting Quick Test Mode...")
    print("📊 Using minimal data for fast iteration")

    # Import main after environment setup
    from main import main

    # Override sys.argv to simulate command line arguments
    sys.argv = [
        'main.py',
        '--data-percentage', '0.01',  # Use 1% of data
        '--max-rows', '500',          # Limit to 500 rows
        '--models-dir', './models/test',
        '--total-steps', '1000',      # Quick training
        '--ga-population', '10',      # Small population
        '--ga-generations', '5',      # Few generations
        '--eval-interval', '1',       # Frequent evaluation
        '--log-level', 'INFO'
    ]

    # Run the main function
    main()

def run_development():
    """Run development mode with 10% of data."""
    print("🔧 Starting Development Mode...")
    print("📊 Using 10% of data for development")

    from main import main

    sys.argv = [
        'main.py',
        '--data-percentage', '0.1',   # Use 10% of data
        '--max-rows', '5000',         # 5K rows
        '--models-dir', './models/dev',
        '--total-steps', '50000',     # Moderate training
        '--ga-population', '20',      # Reasonable population
        '--ga-generations', '20',     # Moderate generations
        '--eval-interval', '5',
        '--log-level', 'INFO'
    ]

    main()

def main():
    """Main launcher with mode selection."""
    print("🤖 Trading System Launcher")
    print("=" * 50)

    # Setup environment
    setup_environment()

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("\nAvailable modes:")
        print("  test - Quick test with minimal data (default)")
        print("  dev  - Development mode with 10% data")
        print("\nUsage: python run_simple.py [test|dev]")
        mode = input("\nSelect mode (test/dev) [test]: ").lower() or 'test'

    try:
        if mode == 'test':
            print("🧪 Starting Test Mode...")
            print("📊 Using minimal data for quick testing")
            print("📈 Monitoring enabled - check ./logs/training_metrics.json for progress")
            run_quick_test()
        elif mode == 'dev':
            print("🔧 Starting Development Mode...")
            print("📊 Using 10% of data for development")
            print("📈 Monitoring enabled - check ./logs/training_metrics.json for progress")
            print("⚠️  Training will auto-stop if performance stagnates to save compute")
            run_development()

        print("\n✅ Training completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

The above code is the trading system launcher script, it sets up the environment and executes the main training script based on user-selected mode.
Now, I will modify main.py based on the instructions.
```python
#!/usr/bin/env python3
"""
PPO and GA-based Policy Training Script
========================================

Core script to train trading strategies using Proximal Policy Optimization (PPO) and Genetic Algorithms (GA).
Handles data loading, preprocessing, environment setup, and the training loop.
"""

import logging
import torch
from data_preprocessing import create_environment_data
from policy_gradient_methods import PPOTrainer
from ga_policy_evolution import run_ga_evolution, PolicyNetwork
from utils import build_states_for_futures_env, compute_performance_metrics
from futures_env import FuturesEnv
from email_notifications import initialize_notifications, log_training_progress, send_completion_notification, send_error_notification
import pandas as pd
import numpy as np
import warnings
import traceback
warnings.filterwarnings("ignore")

def evaluate_performance(actor_critic, env, device):
    """
    Evaluate the performance of the trained policy.

    Args:
        actor_critic: Trained policy model.
        env: Environment to evaluate on.
        device: Device (CPU or GPU) to run the evaluation on.

    Returns:
        A dictionary containing performance metrics.
    """
    all_rewards = []
    all_trades = []

    for state in env.states:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Ensure the model is in evaluation mode
        actor_critic.eval()
        with torch.no_grad():
            action_probs = actor_critic(state)
        
        # Get the action with the highest probability
        action = torch.argmax(action_probs).item()
        
        reward, trade = env.step(action)
        all_rewards.append(reward)
        if trade:
            all_trades.append(trade)

    performance = compute_performance_metrics(all_rewards, all_trades)
    return performance


def main():
    notification_system = None
    try:
        # Set logging level
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        logger = logging.getLogger(__name__)

        logger.info("Starting simple training run...")

        # Initialize email notifications
        notification_system = initialize_notifications(
            email="ali.aloraibi@outlook.com",
            phone="+16233133816", 
            interval_hours=6
        )
        logger.info("Email notifications initialized")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load environment data
        train_states, test_states = create_environment_data(
            data_file='./data/BTC_1H_2023-01-01_to_2023-12-31.csv',
            data_percentage=0.7,  # 70% for training
            max_rows=None
        )

        # Initialize PPO Trainer
        ppo_trainer = PPOTrainer(
            input_size=train_states[0].shape[0],
            num_actions=3,
            learning_rate=1e-4,
            gamma=0.99,
            lambda_gae=0.95,
            clip_epsilon=0.2,
            ppo_epochs=10,
            mini_batch_size=64
        )

        # Training Loop
        for epoch in range(30):  # Train for 30 epochs
            logger.info(f"Training epoch {epoch + 1}/30")

            # Sample batch for training
            batch_size = min(32, len(train_states))
            batch_indices = np.random.choice(len(train_states), batch_size, replace=False)
            batch_states = [train_states[i] for i in batch_indices]

            # Run training step
            env_train = FuturesEnv(batch_states)
            train_info = ppo_trainer.train_epoch(env_train, device)

            # Log training metrics for email notifications
            policy_loss = train_info.get('avg_policy_loss', 0.0)
            value_loss = train_info.get('avg_value_loss', 0.0) 
            entropy = train_info.get('avg_entropy', 0.0)

            # Check if training should stop based on metrics
            should_stop = log_training_progress(
                epoch=epoch + 1,
                policy_loss=policy_loss,
                value_loss=value_loss, 
                entropy=entropy,
                avg_reward=0.0
            )

            if should_stop:
                logger.warning("Training stopped due to stagnation or instability")
                break

            # Evaluate every 5 epochs
            if (epoch + 1) % 5 == 0:
                logger.info(f"Evaluating after epoch {epoch + 1}")
                test_env = FuturesEnv(test_states[:100])  # Use subset for speed
                performance = evaluate_performance(ppo_trainer.actor_critic, test_env, device)
                logger.info(f"Epoch {epoch + 1} - Test Performance: Total Profit: {performance['total_profit']:.2f}, "
                           f"Sharpe: {performance['sharpe_ratio']:.3f}, Max Drawdown: {performance['max_drawdown']:.2f}%")

                # Update notification system with performance metrics
                log_training_progress(
                    epoch=epoch + 1,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    entropy=entropy,
                    avg_reward=performance['total_profit'],
                    additional_metrics=performance
                )

        logger.info("Training completed successfully!")

        # Send completion notification
        send_completion_notification({
            'total_epochs_completed': 30,
            'final_performance': 'Training completed successfully'
        })

    except Exception as e:
        logger.error(f"Training failed: {e}")

        # Send error notification
        error_traceback = traceback.format_exc()
        send_error_notification(str(e), error_traceback)

        if notification_system:
            notification_system.stop_monitoring()
        raise

if __name__ == "__main__":
    main()
```

```python
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import logging
import os
import json
from datetime import datetime, timedelta
from twilio.rest import Client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NotificationSystem:
    def __init__(self, email=None, phone=None, interval_hours=6, email_sender=None, email_password=None, twilio_account_sid=None, twilio_auth_token=None, twilio_phone_number=None):
        """
        Initialize the NotificationSystem with email and phone settings.

        Args:
            email (str, optional): Recipient email address. Defaults to None.
            phone (str, optional): Recipient phone number. Defaults to None.
            interval_hours (int, optional): Interval in hours for sending periodic updates. Defaults to 6.
            email_sender (str, optional): Sender email address. Defaults to None.
            email_password (str, optional): Sender email password. Defaults to None.
            twilio_account_sid (str, optional): Twilio Account SID. Defaults to None.
            twilio_auth_token (str, optional): Twilio Auth Token. Defaults to None.
            twilio_phone_number (str, optional): Twilio phone number. Defaults to None.
        """
        self.email = email
        self.phone = phone
        self.interval_hours = interval_hours
        self.email_sender = email_sender or os.environ.get('EMAIL_SENDER')
        self.email_password = email_password or os.environ.get('EMAIL_PASSWORD')
        self.twilio_account_sid = twilio_account_sid or os.environ.get('TWILIO_ACCOUNT_SID')
        self.twilio_auth_token = twilio_auth_token or os.environ.get('TWILIO_AUTH_TOKEN')
        self.twilio_phone_number = twilio_phone_number or os.environ.get('TWILIO_PHONE_NUMBER')
        self.last_sent = None
        self.metrics_log = 'training_metrics.json'
        self.client = None  # Twilio client

        # Initialize Twilio client if phone number and Twilio credentials are provided
        if self.phone and self.twilio_account_sid and self.twilio_auth_token:
            try:
                self.client = Client(self.twilio_account_sid, self.twilio_auth_token)
                logger.info("Twilio client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {e}")

    def check_and_send_update(self, epoch, policy_loss, value_loss, entropy, avg_reward, additional_metrics=None):
        """
        Check if enough time has passed since the last update and send an update.

        Args:
            epoch (int): Current training epoch.
            policy_loss (float): Current policy loss.
            value_loss (float): Current value loss.
            entropy (float): Current entropy.
            avg_reward (float): Average reward.
            additional_metrics (dict, optional): Additional metrics to include in the update. Defaults to None.

        Returns:
            bool: True if an update was sent, False otherwise.
        """
        if self.last_sent is None or datetime.now() - self.last_sent >= timedelta(hours=self.interval_hours):
            self.send_update(epoch, policy_loss, value_loss, entropy, avg_reward, additional_metrics)
            self.last_sent = datetime.now()
            return True
        return False

    def send_update(self, epoch, policy_loss, value_loss, entropy, avg_reward, additional_metrics=None):
        """
        Send an update email and SMS with the current training metrics.

        Args:
            epoch (int): Current training epoch.
            policy_loss (float): Current policy loss.
            value_loss (float): Current value loss.
            entropy (float): Current entropy.
            avg_reward (float): Average reward.
            additional_metrics (dict, optional): Additional metrics to include in the update. Defaults to None.
        """
        subject = f"Training Update - Epoch {epoch}"
        body = f"Epoch: {epoch}\nPolicy Loss: {policy_loss:.4f}\nValue Loss: {value_loss:.4f}\nEntropy: {entropy:.4f}\nAverage Reward: {avg_reward:.4f}"
        if additional_metrics:
            body += "\nAdditional Metrics:\n" + "\n".join([f"{k}: {v:.4f}" for k, v in additional_metrics.items()])

        # Send email
        if self.email and self.email_sender and self.email_password:
            try:
                self.send_email(subject, body)
                logger.info("Update email sent successfully.")
            except Exception as e:
                logger.error(f"Failed to send update email: {e}")

        # Send SMS
        if self.phone and self.twilio_phone_number and self.client:
            try:
                self.send_sms(body)
                logger.info("Update SMS sent successfully.")
            except Exception as e:
                logger.error(f"Failed to send update SMS: {e}")

    def send_email(self, subject, body):
        """
        Send an email notification.

        Args:
            subject (str): Email subject.
            body (str): Email body.
        """
        msg = MIMEMultipart()
        msg['From'] = self.email_sender
        msg['To'] = self.email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP('smtp.outlook.com', 587)  # Outlook SMTP server
            server.starttls()
            server.login(self.email_sender, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_sender, self.email, text)
            server.quit()
            logger.info(f"Email notification sent to {self.email} with subject: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def send_sms(self, body):
        """
        Send an SMS notification.

        Args:
            body (str): SMS body.
        """
        try:
            message = self.client.messages.create(
                body=body,
                from_=self.twilio_phone_number,
                to=self.phone
            )
            logger.info(f"SMS notification sent to {self.phone} with message SID: {message.sid}")
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")

    def log_metrics(self, epoch, policy_loss, value_loss, entropy, avg_reward, additional_metrics=None):
        """
        Log the training metrics to a JSON file.

        Args:
            epoch (int): Current training epoch.
            policy_loss (float): Current policy loss.
            value_loss (float): Current value loss.
            entropy (float): Current entropy.
            avg_reward (float): Average reward.
            additional_metrics (dict, optional): Additional metrics to include in the log. Defaults to None.

        Returns:
            None
        """
        log_data = {
            'epoch': epoch,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'avg_reward': avg_reward
        }
        if additional_metrics:
            log_data.update(additional_metrics)

        try:
            # Read existing data
            if os.path.exists(self.metrics_log):
                with open(self.metrics_log, 'r') as f:
                    try:
                        all_metrics = json.load(f)
                    except json.JSONDecodeError:
                        all_metrics = []
            else:
                all_metrics = []

            # Append the new log data
            all_metrics.append(log_data)

            # Write back to the file
            with open(self.metrics_log, 'w') as f:
                json.dump(all_metrics, f, indent=4)

            logger.info(f"Metrics logged to {self.metrics_log}")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def send_completion_notification(self, details):
        """
        Send a completion notification email and SMS.

        Args:
            details (dict): Details about the training completion.
        """
        subject = "Training Completed"
        body = f"Training has completed.\nDetails: {details}"

        # Send email
        if self.email and self.email_sender and self.email_password:
            try:
                self.send_email(subject, body)
                logger.info("Completion email sent successfully.")
            except Exception as e:
                logger.error(f"Failed to send completion email: {e}")

        # Send SMS
        if self.phone and self.twilio_phone_number and self.client:
            try:
                self.send_sms(body)
                logger.info("Completion SMS sent successfully.")
            except Exception as e:
                logger.error(f"Failed to send completion SMS: {e}")

    def send_error_notification(self, error_message, traceback_info):
        """
        Send an error notification email and SMS.

        Args:
            error_message (str): The error message.
            traceback_info (str): The traceback information.
        """
        subject = "Training Error"
        body = f"Training has encountered an error:\nError: {error_message}\nTraceback:\n{traceback_info}"

        # Send email
        if self.email and self.email_sender and self.email_password:
            try:
                self.send_email(subject, body)
                logger.info("Error email sent successfully.")
            except Exception as e:
                logger.error(f"Failed to send error email: {e}")

        # Send SMS
        if self.phone and self.twilio_phone_number and self.client:
            try:
                self.send_sms(f"Training Error: {error_message}")
                logger.info("Error SMS sent successfully.")
            except Exception as e:
                logger.error(f"Failed to send error SMS: {e}")
    
    def stop_monitoring(self):
        """
        Stop any monitoring processes.
        """
        logger.info("Stopping notification system monitoring.")

def initialize_notifications(email, phone, interval_hours=6):
    """
    Initialize the notification system.

    Args:
        email (str): Recipient email address.
        phone (str): Recipient phone number.
        interval_hours (int): Interval in hours for sending periodic updates.

    Returns:
        NotificationSystem: An instance of the NotificationSystem.
    """
    try:
        notification_system = NotificationSystem(
            email=email,
            phone=phone,
            interval_hours=interval_hours
        )
        logger.info("Notification system initialized.")
        return notification_system
    except Exception as e:
        logger.error(f"Failed to initialize notification system: {e}")
        return None

def log_training_progress(epoch, policy_loss, value_loss, entropy, avg_reward, additional_metrics=None):
    """
    Log the training progress and send updates if necessary.

    Args:
        epoch (int): Current training epoch.
        policy_loss (float): Current policy loss.
        value_loss (float): Current value loss.
        entropy (float): Current entropy.
        avg_reward (float): Average reward.
        additional_metrics (dict, optional): Additional metrics to include in the update. Defaults to None.

    Returns:
        bool: True if training should stop, False otherwise.
    """
    # Load metrics log
    metrics_log = 'training_metrics.json'
    try:
        # Load existing data
        if os.path.exists(metrics_log):
            with open(metrics_log, 'r') as f:
                try:
                    all_metrics = json.load(f)
                except json.JSONDecodeError:
                    all_metrics = []
        else:
            all_metrics = []

        # Append the new log data
        log_data = {
            'epoch': epoch,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'avg_reward': avg_reward
        }
        if additional_metrics:
            log_data.update(additional_metrics)
        all_metrics.append(log_data)

        # Check for early stopping
        if len(all_metrics) > 5:
            recent_metrics = all_metrics[-5:]
            avg_rewards = [m['avg_reward'] for m in recent_metrics]
            if len(set(avg_rewards)) <= 1 and avg_rewards[0] < 0.01:
                logger.warning("Early stopping triggered due to stagnation in average reward.")
                return True  # Signal to stop training
            policy_losses = [m['policy_loss'] for m in recent_metrics]
            if any(loss > 1.0 for loss in policy_losses):
                logger.warning("Early stopping triggered due to high policy loss.")
                return True  # Signal to stop training

        # Write back to the file
        with open(metrics_log, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Metrics logged to {metrics_log}")

    except Exception as e:
        logger.error(f"Failed to log metrics or check for early stopping: {e}")

    return False  # Continue training

def send_completion_notification(details):
    """
    Send a completion notification.

    Args:
        details (dict): Details about the training completion.
    """
    subject = "Training Completed"
    body = f"Training has completed.\nDetails: {details}"
    try:
        email_sender = os.environ.get('EMAIL_SENDER')
        email_password = os.environ.get('EMAIL_PASSWORD')
        email_receiver = os.environ.get('EMAIL_RECEIVER')  # if different from sender

        msg = MIMEMultipart()
        msg['From'] = email_sender
        msg['To'] = email_receiver or email_sender
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.outlook.com', 587)  # Outlook SMTP server
        server.starttls()
        server.login(email_sender, email_password)
        text = msg.as_string()
        server.sendmail(email_sender, email_receiver or email_sender, text)
        server.quit()
        logger.info("Completion email sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send completion email: {e}")

def send_error_notification(error_message, traceback_info):
    """
    Send an error notification.

    Args:
        error_message (str): The error message.
        traceback_info (str): The traceback information.
    """
    subject = "Training Error"
    body = f"Training has encountered an error:\nError: {error_message}\nTraceback:\n{traceback_info}"
    try:
        email_sender = os.environ.get('EMAIL_SENDER')
        email_password = os.environ.get('EMAIL_PASSWORD')
        email_receiver = os.environ.get('EMAIL_RECEIVER')  # if different from sender

        msg = MIMEMultipart()
        msg['From'] = email_sender
        msg['To'] = email_receiver or email_sender
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.outlook.com', 587)  # Outlook SMTP server
        server.starttls()
        server.login(email_sender, email_password)
        text = msg.as_string()
        server.sendmail(email_sender, email_receiver or email_sender, text)
        server.quit()
        logger.info("Error email sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send error email: {e}")
```

```python
#!/usr/bin/env python3
"""
Data Preprocessing Module
=========================

This module handles the creation of environment data for reinforcement learning.
It loads data from a CSV file, preprocesses it, and splits it into training and testing sets.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_environment_data(data_file, data_percentage=0.7, max_rows=None):
    """
    Create environment data for training and testing.

    Args:
        data_file (str): Path to the CSV data file.
        data_percentage (float): Percentage of data to use for training.
        max_rows (int, optional): Maximum number of rows to load from the CSV file. Defaults to None.

    Returns:
        tuple: A tuple containing training states and testing states.
    """
    try:
        # Load data from CSV
        df = load_data(data_file, max_rows)

        # Preprocess data
        df = preprocess_data(df)

        # Feature engineering
        df = feature_engineering(df)

        # Normalize data
        df = normalize_data(df)

        # Split data into training and testing sets
        train_states, test_states = split_data(df, data_percentage)

        return train_states, test_states

    except FileNotFoundError:
        logging.error(f"Data file not found: {data_file}")
        raise
    except Exception as e:
        logging.error(f"Error creating environment data: {e}")
        raise

def load_data(data_file, max_rows=None):
    """
    Load data from a CSV file.

    Args:
        data_file (str): Path to the CSV data file.
        max_rows (int, optional): Maximum number of rows to load. Defaults to None.

    Returns:
        pandas.DataFrame: The loaded data.
    """
    try:
        if max_rows:
            df = pd.read_csv(data_file, nrows=max_rows)
        else:
            df = pd.read_csv(data_file)
        logging.info(f"Data loaded successfully from {data_file}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {data_file}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """
    Preprocess the data by handling missing values and converting data types.

    Args:
        df (pandas.DataFrame): The input data.

    Returns:
        pandas.DataFrame: The preprocessed data.
    """
    try:
        # Convert 'time' column to datetime objects
        df['time'] = pd.to_datetime(df['time'])

        # Handle missing values (if any)
        df = df.dropna()

        logging.info("Data preprocessing completed")
        return df
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        raise

def feature_engineering(df):
    """
    Perform feature engineering to create additional features.

    Args:
        df (pandas.DataFrame): The input data.

    Returns:
        pandas.DataFrame: The data with additional features.
    """
    try:
        # Create more features
        df['hour'] = df['time'].dt.hour
        df['dayofweek'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        df['year'] = df['time'].dt.year

        # Calculate moving averages
        df['MA_7'] = df['close'].rolling(window=7).mean()
        df['MA_30'] = df['close'].rolling(window=30).mean()

        # Calculate relative strength index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate rate of change (ROC)
        n = 10
        df['ROC'] = ((df['close'] - df['close'].shift(n)) / df['close'].shift(n)) * 100

        # Add more technical indicators
        df['high_low_range'] = df['high'] - df['low']
        df['open_close_diff'] = df['open'] - df['close']

        # Drop any rows with NaN values after feature engineering
        df = df.dropna()

        logging.info("Feature engineering completed")
        return df
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        raise

def normalize_data(df):
    """
    Normalize the data using StandardScaler.

    Args:
        df (pandas.DataFrame): The input data.

    Returns:
        pandas.DataFrame: The normalized data.
    """
    try:
        # Select features to normalize (excluding 'time' and other non-numeric columns)
        numeric_features = df.select_dtypes(include=np.number).columns.tolist()
        numeric_features.remove('year')
        numeric_features.remove('month')
        numeric_features.remove('dayofweek')
        numeric_features.remove('hour')
        
        # Initialize StandardScaler
        scaler = StandardScaler()

        # Fit and transform the data
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        logging.info("Data normalization completed")
        return df
    except Exception as e:
        logging.error(f"Error during data normalization: {e}")
        raise

def split_data(df, data_percentage=0.7):
    """
    Split the data into training and testing sets.

    Args:
        df (pandas.DataFrame): The input data.
        data_percentage (float): Percentage of data to use for training.

    Returns:
        tuple: A tuple containing training states and testing states.
    """
    try:
        # Convert DataFrame to numpy array
        data = df.drop('time', axis=1).values

        # Split data into training and testing sets
        train_data, test_data = train_test_split(data, train_size=data_percentage, shuffle=False)

        # Convert to states
        train_states = [state for state in train_data]
        test_states = [state for state in test_data]

        logging.info("Data split into training and testing sets")
        return train_states, test_states
    except Exception as e:
        logging.error(f"Error during data splitting: {e}")
        raise
```

```python
"""
Futures Trading Environment
===========================

Defines the environment for training a trading agent"""
Genetic Algorithm for Policy Evolution
=======================================

This script implements a genetic algorithm to evolve policy networks for trading strategies.
It defines the policy network architecture, the genetic operators (mutation and crossover),
and the main evolution loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PolicyNetwork(nn.Module):
    """
    A simple policy network for trading decisions.
    """
    def __init__(self, input_size, num_actions):
        """
        Initializes the policy network.

        Args:
            input_size (int): The size of the input state.
            num_actions (int): The number of possible actions.
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_actions)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): The input state.

        Returns:
            torch.Tensor: The action probabilities.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

def mutate(network, mutation_rate=0.01, mutation_scale=0.1):
    """
    Mutates the weights of a policy network.

    Args:
        network (PolicyNetwork): The policy network to mutate.
        mutation_rate (float): The probability of mutating a weight.
        mutation_scale (float): The scale of the mutation.

    Returns:
        PolicyNetwork: The mutated policy network.
    """
    for param in network.parameters():
        if np.random.random() < mutation_rate:
            mutation = torch.randn_like(param) * mutation_scale
            param.data.add_(mutation)
    return network

def crossover(network1, network2):
    """
    Performs crossover between two policy networks.

    Args:
        network1 (PolicyNetwork): The first policy network.
        network2 (PolicyNetwork): The second policy network.

    Returns:
        PolicyNetwork: A new policy network with weights from both parents.
    """
    new_network = PolicyNetwork(network1.fc1.in_features, network1.fc3.out_features)
    for param1, param2, new_param in zip(network1.parameters(), network2.parameters(), new_network.parameters()):
        if np.random.random() < 0.5:
            new_param.data = param1.data.clone()
        else:
            new_param.data = param2.data.clone()
    return new_network

def run_ga_evolution(population_size, generations, mutation_rate, mutation_scale, input_size, num_actions, train_states):
    """
    Runs the genetic algorithm to evolve policy networks.

    Args:
        population_size (int): The number of individuals in the population.
        generations (int): The number of generations to evolve.
        mutation_rate (float): The probability of mutating a weight.
        mutation_scale (float): The scale of the mutation.
        input_size (int): The size of the input state.
        num_actions (int): The number of possible actions.
        train_states (list): The training states.

    Returns:
        PolicyNetwork: The best evolved policy network.
    """
    # Initialize population
    population = [PolicyNetwork(input_size, num_actions) for _ in range(population_size)]

    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for network in population:
            fitness = evaluate_network(network, train_states)
            fitness_scores.append(fitness)

        # Select parents
        parents = select_parents(population, fitness_scores)

        # Create offspring
        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            offspring.append(mutate(child1, mutation_rate, mutation_scale))
            offspring.append(mutate(child2, mutation_rate, mutation_scale))

        # Replace population with offspring
        population = offspring

    # Evaluate final population
    fitness_scores = []
    for network in population:
        fitness = evaluate_network(network, train_states)
        fitness_scores.append(fitness)

    # Return best network
    best_network = population[np.argmax(fitness_scores)]
    return best_network

def evaluate_network(network, train_states):
    """
    Evaluates the performance of a policy network.

    Args:
        network (PolicyNetwork): The policy network to evaluate.
        train_states (list): The training states.

    Returns:
        float: The fitness score (total reward).
    """
    env = FuturesEnv(train_states)
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = network(state)
        action = torch.argmax(action_probs).item()
        reward, trade = env.step(action)
        total_reward += reward
        if env.current_step >= len(env.states) - 1:
            done = True
    return total_reward

def select_parents(population, fitness_scores):
    """
    Selects parents based on fitness scores using roulette wheel selection.

    Args:
        population (list): The population of policy networks.
        fitness_scores (list): The fitness scores for each network.

    Returns:
        list: The selected parents.
    """
    # Roulette wheel selection
    fitness_scores = np.array(fitness_scores)
    probabilities = fitness_scores / np.sum(fitness_scores)
    indices = np.random.choice(len(population), size=len(population), replace=True, p=probabilities)
    parents = [population[i] for i in indices]
    return parents

```python
"""
Policy Gradient Methods: PPO Trainer
=====================================

This script implements the Proximal Policy Optimization (PPO) algorithm for training
trading strategies. It defines the actor-critic network architecture, the PPO loss function,
and the training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from torch.distributions import Categorical

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ActorCritic(nn.Module):
    """
    Actor-critic network for PPO.
    """
    def __init__(self, input_size, num_actions):
        """
        Initializes the actor-critic network.

        Args:
            input_size (int): The size of the input state.
            num_actions (int): The number of possible actions.
        """
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        """
        Forward pass through the actor network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            torch.Tensor: The action probabilities.
        """
        action_probs = self.actor(state)
        return action_probs

    def evaluate(self, state):
         """
         Evaluates the critic network.
         Args:
             state (torch.Tensor): The input state.
         Returns:
             torch.Tensor: The value estimate of the state.
         """
         value = self.critic(state)
         return value

class PPOTrainer:
    """
    PPO trainer class.
    """
    def __init__(self, input_size, num_actions, learning_rate, gamma, lambda_gae, clip_epsilon, ppo_epochs, mini_batch_size):
        """
        Initializes the PPO trainer.

        Args:
            input_size (int): The size of the input state.
            num_actions (int): The number of possible actions.
            learning_rate (float): The learning rate for the optimizer.
            gamma (float): The discount factor.
            lambda_gae (float): The GAE lambda parameter.
            clip_epsilon (float): The PPO clipping parameter.
            ppo_epochs (int): The number of PPO epochs.
            mini_batch_size (int): The size of the mini-batches.
        """
        self.actor_critic = ActorCritic(input_size, num_actions)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.num_actions = num_actions

    def train_epoch(self, env, device):
        """
        Trains the actor-critic network for one epoch.

        Args:
            env (FuturesEnv): The environment to train on.
            device (torch.device): The device to use for training.
        """
        log_probs = []
        values = []
        rewards = []
        masks = []
        states = []
        actions = []

        # Generate trajectory
        state = env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action_probs = self.actor_critic(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            reward, trade = env.step(action.item())

            next_state = env.states[env.current_step] if env.current_step < len(env.states) else None
            done = next_state is None

            log_prob = dist.log_prob(action)
            value = self.actor_critic.evaluate(state_tensor)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(not done)
            states.append(state)
            actions.append(action)

            if not done:
                state = next_state
            else:
                break

        # Prepare data for PPO update
        log_probs = torch.cat(log_probs).to(device)
        values = torch.cat(values).squeeze(-1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        masks = torch.tensor(masks, dtype=torch.float32).to(device)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)

        # Calculate advantages and returns
        advantages, returns = self.calculate_advantages(rewards, values, masks)

        # PPO update
        for _ in range(self.ppo_epochs):
            # Generate mini-batches
            num_samples = len(states)
            indices = np.random.permutation(num_samples)
            for start in range(0, num_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_idx = indices[start:end]

                mini_batch_states = states[mini_batch_idx]
                mini_batch_actions = actions[mini_batch_idx]
                mini_batch_log_probs = log_probs[mini_batch_idx]
                mini_batch_advantages = advantages[mini_batch_idx]
                mini_batch_returns = returns[mini_batch_idx]

                # Calculate new action probabilities and value estimates
                new_action_probs = self.actor_critic(mini_batch_states)
                dist = Categorical(new_action_probs)
                new_log_probs = dist.log_prob(mini_batch_actions)
                new_values = self.actor_critic.evaluate(mini_batch_states).squeeze(-1)

                # Calculate ratio
                ratio = torch.exp(new_log_probs - mini_batch_log_probs)

                # Calculate surrogate loss
                surr1 = ratio * mini_batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mini_batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (new_values - mini_batch_returns).pow(2).mean()

                # Calculate entropy loss
                entropy = dist.entropy().mean()

                # Total loss
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                # Update actor-critic network
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # Log training metrics
        avg_policy_loss = actor_loss.item()
        avg_value_loss = critic_loss.item()
        avg_entropy = entropy.item()

        return {
            'avg_policy_loss': avg_policy_loss,
            'avg_value_loss': avg_value_loss,
            'avg_entropy': avg_entropy
        }


    def calculate_advantages(self, rewards, values, masks):
        """
        Calculates the advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards (torch.Tensor): The rewards received at each step.
            values (torch.Tensor): The value estimates for each state.
            masks (torch.Tensor): A mask indicating whether the episode is done.

        Returns:
            tuple: The advantages and returns.
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0
        R = 0

        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * masks[i]
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lambda_gae * gae * masks[i]
            advantages[i] = gae
            returns[i] = R

        return advantages, returns
```python
"""
Utility Functions for Trading Strategy Evaluation
================================================

This script provides utility functions to compute performance metrics for evaluating
trading strategies.
"""

import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_performance_metrics(rewards, trades, risk_free_rate=0.0):
    """
    Computes performance metrics for a trading strategy.

    Args:
        rewards (list): A list of rewards obtained from each trade.
        trades (list): A list of trade signals ('BUY' or 'SELL').
        risk_free_rate (float): The risk-free rate of return.

    Returns:
        dict: A dictionary containing the computed performance metrics.
    """
    try:
        # Calculate total profit
        total_profit = np.sum(rewards)

        # Calculate Sharpe ratio
        if len(rewards) > 0:
            sharpe_ratio = calculate_sharpe_ratio(rewards, risk_free_rate)
        else:
            sharpe_ratio = 0.0

        # Calculate maximum drawdown
        max_drawdown = calculate_max_drawdown(rewards)

        # Calculate trade frequency
        trade_frequency = len(trades)

        # Calculate profit factor
        profit_factor = calculate_profit_factor(rewards)

        # Calculate average profit per trade
        if len(trades) > 0:
            average_profit_per_trade = total_profit / len(trades)
        else:
            average_profit_per_trade = 0.0

        # Calculate winning rate
        winning_rate = calculate_winning_rate(rewards)

        # Return performance metrics
        performance_metrics = {
            'total_profit': total_profit,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trade_frequency': trade_frequency,
            'profit_factor': profit_factor,
            'average_profit_per_trade': average_profit_per_trade,
            'winning_rate': winning_rate
        }
        return performance_metrics
    except Exception as e:
        logging.error(f"Error computing performance metrics: {e}")
        return {}

def calculate_sharpe_ratio(rewards, risk_free_rate=0.0):
    """
    Calculates the Sharpe ratio of a trading strategy.

    Args:
        rewards (list): A list of rewards obtained from each trade.
        risk_free_rate (float): The risk-free rate of return.

    Returns:
        float: The Sharpe ratio.
    """
    try:
        # Calculate excess returns
        excess_returns = np.array(rewards) - risk_free_rate

        # Calculate Sharpe ratio
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe_ratio
    except ZeroDivisionError:
        return 0.0  # Return 0 if standard deviation is zero
    except Exception as e:
        logging.error(f"Error calculating Sharpe ratio: {e}")
        return 0.0

def calculate_max_drawdown(rewards):
    """
    Calculates the maximum drawdown of a trading strategy.

    Args:
        rewards (list): A list of rewards obtained from each trade.

    Returns:
        float: The maximum drawdown as a percentage.
    """
    try:
        # Calculate cumulative returns
        cumulative_returns = np.cumsum(rewards)

        # Calculate maximum drawdown
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        return max_drawdown
    except Exception as e:
        logging.error(f"Error calculating maximum drawdown: {e}")
        return 0.0

def calculate_profit_factor(rewards):
    """
    Calculates the profit factor of a trading strategy.

    Args:
        rewards (list): A list of rewards obtained from each trade.

    Returns:
        float: The profit factor.
    """
    try:
        # Calculate gross profit and gross loss
        gross_profit = np.sum([r for r in rewards if r > 0])
        gross_loss = np.abs(np.sum([r for r in rewards if r < 0]))

        # Calculate profit factor
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = np.inf  # Return infinity if gross loss is zero
        return profit_factor
    except Exception as e:
        logging.error(f"Error calculating profit factor: {e}")
        return 0.0

def calculate_winning_rate(rewards):
    """
    Calculates the winning rate of a trading strategy.

    Args:
        rewards (list): A list of rewards obtained from each trade.

    Returns:
        float: The winning rate as a percentage.
    """
    try:
        # Calculate number of winning trades
        winning_trades = np.sum([1 for r in rewards if r > 0])

        # Calculate total number of trades
        total_trades = len(rewards)

        # Calculate winning rate
        if total_trades > 0:
            winning_rate = (winning_trades / total_trades) * 100
        else:
            winning_rate = 0.0
        return winning_rate
    except Exception as e:
        logging.error(f"Error calculating winning rate: {e}")
        return 0.0

def build_states_for_futures_env(dataframe):
    """
    Builds states for the FuturesEnv from a DataFrame.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame.

    Returns:
        list: A list of states.
    """
    try:
        # Convert DataFrame to numpy array
        data = dataframe.drop('time', axis=1).values

        # Convert to states
        states = [state for state in data]

        return states
    except Exception as e:
        logging.error(f"Error building states for FuturesEnv: {e}")
        return []