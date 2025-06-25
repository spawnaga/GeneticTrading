
#!/usr/bin/env python
"""
Real-time Training Monitoring Dashboard
======================================

Provides comprehensive monitoring of training progress with early warning systems
for ineffective training runs to save compute resources.
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from email_notifications import TrainingNotificationSystem

class TrainingMonitor:
    """Real-time training monitor with early stopping recommendations."""
    
    def __init__(self, log_dir: str = "./logs", models_dir: str = "./models", enable_email: bool = True):
        self.log_dir = Path(log_dir)
        self.models_dir = Path(models_dir)
        self.metrics_file = self.log_dir / "training_metrics.json"
        self.performance_history = []
        self.method_switches = []
        self.warning_flags = []
        
        # Thresholds for early stopping recommendations
        self.min_iterations_before_warning = 5
        self.stagnation_threshold = 10  # iterations without improvement
        self.performance_drop_threshold = 0.5  # 50% performance drop
        self.time_per_iteration_limit = 600  # 10 minutes max per iteration
        
        # Create monitoring directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Email notification setup
        self.email_manager = None
        if enable_email:
            self.email_manager = TrainingNotificationSystem()
            if hasattr(self.email_manager, 'setup_from_config') and self.email_manager.setup_from_config():
                self.email_manager.start_notifications()
                logging.info("Email notifications enabled - reports every 6 hours")
            else:
                logging.info("Email notifications not configured - check ./config/email_config.json")
        
    def log_iteration(self, iteration: int, method: str, performance: float, 
                     metrics: Dict, training_time: float, switch_reason: str = ""):
        """Log a training iteration with comprehensive metrics."""
        timestamp = datetime.now().isoformat()
        
        entry = {
            "timestamp": timestamp,
            "iteration": iteration,
            "method": method,
            "performance": performance,
            "training_time": training_time,
            "switch_reason": switch_reason,
            "metrics": metrics
        }
        
        self.performance_history.append(entry)
        
        # Save to file
        self._save_metrics()
        
        # Check for warning conditions
        warnings = self._check_warning_conditions(entry)
        if warnings:
            self.warning_flags.extend(warnings)
            self._log_warnings(warnings)
            
        # Print progress summary
        self._print_progress_summary(entry, warnings)
        
    def _check_warning_conditions(self, entry: Dict) -> List[str]:
        """Check for conditions that suggest training should be stopped."""
        warnings = []
        iteration = entry["iteration"]
        performance = entry["performance"]
        training_time = entry["training_time"]
        
        if iteration < self.min_iterations_before_warning:
            return warnings
            
        # Check for extended stagnation
        if len(self.performance_history) >= self.stagnation_threshold:
            recent_performances = [h["performance"] for h in self.performance_history[-self.stagnation_threshold:]]
            best_recent = max(recent_performances)
            current_best = max([h["performance"] for h in self.performance_history])
            
            if best_recent < current_best * 0.95:  # No improvement in recent iterations
                warnings.append(f"STAGNATION: No improvement in last {self.stagnation_threshold} iterations")
                
        # Check for performance collapse
        if len(self.performance_history) >= 3:
            recent_avg = np.mean([h["performance"] for h in self.performance_history[-3:]])
            historical_avg = np.mean([h["performance"] for h in self.performance_history[:-3]])
            
            if recent_avg < historical_avg * self.performance_drop_threshold:
                warnings.append(f"PERFORMANCE_COLLAPSE: Recent performance dropped {(1-recent_avg/historical_avg)*100:.1f}%")
                
        # Check for excessive training time per iteration
        if training_time > self.time_per_iteration_limit:
            warnings.append(f"SLOW_TRAINING: Iteration took {training_time/60:.1f} minutes (limit: {self.time_per_iteration_limit/60:.1f})")
            
        # Check for method switching frequency
        if len(self.performance_history) >= 5:
            recent_methods = [h["method"] for h in self.performance_history[-5:]]
            if len(set(recent_methods)) >= 3:
                warnings.append("EXCESSIVE_SWITCHING: Too many method switches in recent iterations")
                
        return warnings
        
    def _log_warnings(self, warnings: List[str]):
        """Log warnings to file and console."""
        warning_file = self.log_dir / "training_warnings.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(warning_file, "a") as f:
            for warning in warnings:
                f.write(f"[{timestamp}] {warning}\n")
                print(f"⚠️  WARNING: {warning}")
                
    def _print_progress_summary(self, entry: Dict, warnings: List[str]):
        """Print a concise progress summary."""
        iteration = entry["iteration"]
        method = entry["method"]
        performance = entry["performance"]
        training_time = entry["training_time"]
        metrics = entry["metrics"]
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration:2d} | Method: {method:3s} | Time: {training_time/60:5.1f}m")
        print(f"Performance: {performance:8.4f} | CAGR: {metrics.get('cagr', 0):6.2f}% | Sharpe: {metrics.get('sharpe', 0):5.2f}")
        
        if warnings:
            print(f"⚠️  {len(warnings)} warning(s) detected!")
            
        # Show recommendation
        recommendation = self.get_training_recommendation()
        if recommendation != "CONTINUE":
            print(f"🚨 RECOMMENDATION: {recommendation}")
            
        print(f"{'='*60}")
        
    def get_training_recommendation(self) -> str:
        """Get recommendation for whether to continue training."""
        if len(self.performance_history) < self.min_iterations_before_warning:
            return "CONTINUE"
            
        # Count severe warnings
        severe_warnings = len([w for w in self.warning_flags if any(keyword in w for keyword in 
                              ["STAGNATION", "PERFORMANCE_COLLAPSE", "EXCESSIVE_SWITCHING"])])
                              
        if severe_warnings >= 3:
            return "STOP_IMMEDIATELY"
        elif severe_warnings >= 2:
            return "CONSIDER_STOPPING"
        elif len(self.warning_flags) >= 5:
            return "REVIEW_HYPERPARAMETERS"
        else:
            return "CONTINUE"
            
    def _save_metrics(self):
        """Save metrics to JSON file."""
        with open(self.metrics_file, "w") as f:
            json.dump({
                "performance_history": self.performance_history,
                "warning_flags": self.warning_flags,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
            
    def generate_progress_report(self) -> str:
        """Generate a comprehensive progress report."""
        if not self.performance_history:
            return "No training data available."
            
        total_iterations = len(self.performance_history)
        total_time = sum(h["training_time"] for h in self.performance_history)
        best_performance = max(h["performance"] for h in self.performance_history)
        current_performance = self.performance_history[-1]["performance"]
        
        # Method usage statistics
        methods = [h["method"] for h in self.performance_history]
        method_counts = {method: methods.count(method) for method in set(methods)}
        
        # Performance trend
        if len(self.performance_history) >= 5:
            recent_trend = np.polyfit(range(5), 
                                    [h["performance"] for h in self.performance_history[-5:]], 1)[0]
            trend_direction = "📈 Improving" if recent_trend > 0 else "📉 Declining"
        else:
            trend_direction = "📊 Insufficient data"
            
        report = f"""
🤖 TRAINING PROGRESS REPORT
{'='*50}
📊 Overall Statistics:
   • Total Iterations: {total_iterations}
   • Total Training Time: {total_time/3600:.1f} hours
   • Average Time/Iteration: {total_time/total_iterations/60:.1f} minutes
   
🎯 Performance Metrics:
   • Best Performance: {best_performance:.4f}
   • Current Performance: {current_performance:.4f}
   • Performance Efficiency: {current_performance/best_performance*100:.1f}%
   • Recent Trend (5 iter): {trend_direction}
   
🔄 Method Usage:
{chr(10).join([f'   • {method}: {count} iterations ({count/total_iterations*100:.1f}%)' 
              for method, count in method_counts.items()])}

⚠️  Warning Summary:
   • Total Warnings: {len(self.warning_flags)}
   • Recommendation: {self.get_training_recommendation()}
   
📈 Next Steps:
{self._get_next_steps_recommendation()}
"""
        return report
        
    def _get_next_steps_recommendation(self) -> str:
        """Get specific recommendations for next steps."""
        recommendation = self.get_training_recommendation()
        
        if recommendation == "STOP_IMMEDIATELY":
            return "   • Stop training immediately to save compute resources\n   • Review hyperparameters and training setup"
        elif recommendation == "CONSIDER_STOPPING":
            return "   • Consider stopping if performance doesn't improve in next 2-3 iterations\n   • Monitor closely for improvements"
        elif recommendation == "REVIEW_HYPERPARAMETERS":
            return "   • Review learning rates, batch sizes, and other hyperparameters\n   • Consider adjusting adaptive switching thresholds"
        else:
            return "   • Continue training - progress looks good\n   • Monitor for sustained improvement"
            
    def create_live_dashboard(self, refresh_interval: int = 30):
        """Create a live updating dashboard."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        def update_plots(frame):
            if not self.performance_history:
                return
                
            # Clear axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
                
            # Performance over time
            iterations = [h["iteration"] for h in self.performance_history]
            performances = [h["performance"] for h in self.performance_history]
            ax1.plot(iterations, performances, 'b-o', linewidth=2, markersize=4)
            ax1.set_title("Performance Over Time")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Performance Score")
            ax1.grid(True, alpha=0.3)
            
            # Method usage over time
            methods = [h["method"] for h in self.performance_history]
            method_colors = {"GA": "red", "PPO": "blue"}
            colors = [method_colors.get(m, "gray") for m in methods]
            ax2.scatter(iterations, [1]*len(iterations), c=colors, s=50)
            ax2.set_title("Training Method Over Time")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Method")
            ax2.set_yticks([1])
            ax2.set_yticklabels(["Method"])
            
            # Training time per iteration
            training_times = [h["training_time"]/60 for h in self.performance_history]  # Convert to minutes
            ax3.bar(iterations, training_times, alpha=0.7)
            ax3.set_title("Training Time per Iteration")
            ax3.set_xlabel("Iteration")
            ax3.set_ylabel("Time (minutes)")
            ax3.grid(True, alpha=0.3)
            
            # Performance metrics comparison
            if len(self.performance_history) > 0:
                latest = self.performance_history[-1]["metrics"]
                metrics_names = ["CAGR", "Sharpe", "MDD"]
                metrics_values = [latest.get("cagr", 0), latest.get("sharpe", 0), -latest.get("mdd", 0)]
                bars = ax4.bar(metrics_names, metrics_values, color=['green', 'blue', 'red'])
                ax4.set_title("Latest Performance Metrics")
                ax4.set_ylabel("Value")
                
                # Add value labels on bars
                for bar, value in zip(bars, metrics_values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
                            
            plt.tight_layout()
            
        # Create animation
        ani = FuncAnimation(fig, update_plots, interval=refresh_interval*1000, blit=False)
        plt.show()
        return ani

# Integration with existing training code
class MonitoredAdaptiveTrainer:
    """Wrapper for AdaptiveTrainer with integrated monitoring."""
    
    def __init__(self, adaptive_trainer, monitor: TrainingMonitor):
        self.trainer = adaptive_trainer
        self.monitor = monitor
        
    def adaptive_train_with_monitoring(self, max_iterations: int = 50):
        """Run adaptive training with real-time monitoring."""
        print("🚀 Starting monitored adaptive training...")
        self.monitor.generate_progress_report()
        
        for iteration in range(max_iterations):
            start_time = time.time()
            
            # Run one iteration of adaptive training
            try:
                # This would integrate with your existing adaptive trainer
                performance, entropy, metrics = self.trainer.evaluate_current_policy()
                method = self.trainer.current_method
                
                # Execute training phase based on current method
                if method == "GA":
                    self.trainer.run_ga_phase()
                else:
                    self.trainer.run_ppo_phase()
                    
                training_time = time.time() - start_time
                
                # Log to monitor
                self.monitor.log_iteration(
                    iteration + 1, method, performance, metrics, training_time
                )
                
                # Check if we should stop based on monitoring
                recommendation = self.monitor.get_training_recommendation()
                if recommendation in ["STOP_IMMEDIATELY", "CONSIDER_STOPPING"]:
                    print(f"\n🛑 Training stopped based on monitor recommendation: {recommendation}")
                    break
                    
            except Exception as e:
                print(f"❌ Error in iteration {iteration + 1}: {e}")
                break
                
        # Generate final report
        print("\n" + self.monitor.generate_progress_report())
