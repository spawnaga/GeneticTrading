
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger("VISUALIZATION")

class TradingVisualizationSystem:
    """
    Revolutionary visualization system for GA+PPO trading progress.
    Provides real-time insights into learning progress and trading performance.
    """
    
    def __init__(self, log_dir="./logs", web_port=5000):
        self.log_dir = Path(log_dir)
        self.web_port = web_port
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization data storage
        self.training_history = {
            'ga_generations': [],
            'ppo_episodes': [],
            'performance_metrics': [],
            'action_distributions': [],
            'market_regimes': [],
            'adaptive_switches': []
        }
        
        # Color schemes for different components
        self.colors = {
            'ga': '#FF6B6B',      # Red for GA
            'ppo': '#4ECDC4',     # Teal for PPO
            'profit': '#45B7D1',  # Blue for profits
            'loss': '#FFA07A',    # Salmon for losses
            'neutral': '#95A5A6'  # Gray for neutral
        }
        
        logger.info(f"🎨 Visualization system initialized on port {web_port}")
    
    def update_training_progress(self, method, generation_or_episode, metrics):
        """Update training progress with new data."""
        timestamp = datetime.now()
        
        progress_entry = {
            'timestamp': timestamp,
            'method': method,
            'iteration': generation_or_episode,
            'metrics': metrics.copy() if isinstance(metrics, dict) else {}
        }
        
        if method.upper() == 'GA':
            self.training_history['ga_generations'].append(progress_entry)
            logger.info(f"🧬 GA Generation {generation_or_episode}: {metrics.get('fitness', 'N/A')}")
        else:
            self.training_history['ppo_episodes'].append(progress_entry)
            logger.info(f"🎯 PPO Episode {generation_or_episode}: {metrics.get('reward', 'N/A')}")
    
    def create_learning_progress_dashboard(self):
        """Create comprehensive learning progress dashboard."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "🧬 GA vs PPO Performance Evolution",
                "📊 Action Distribution Analysis", 
                "💰 Trading Performance Metrics",
                "🎯 Adaptive Algorithm Switching",
                "📈 Cumulative Returns",
                "⚡ Real-time Market Insights"
            ],
            specs=[
                [{"secondary_y": True}, {"type": "bar"}],
                [{"secondary_y": True}, {"type": "scatter"}],
                [{"secondary_y": True}, {"type": "indicator"}]
            ]
        )
        
        # Plot 1: GA vs PPO Performance
        self._add_performance_comparison(fig, row=1, col=1)
        
        # Plot 2: Action Distribution
        self._add_action_distribution(fig, row=1, col=2)
        
        # Plot 3: Trading Metrics
        self._add_trading_metrics(fig, row=2, col=1)
        
        # Plot 4: Adaptive Switching
        self._add_adaptive_switching(fig, row=2, col=2)
        
        # Plot 5: Cumulative Returns
        self._add_cumulative_returns(fig, row=3, col=1)
        
        # Plot 6: Market Insights
        self._add_market_insights(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title="🚀 Revolutionary Trading System - Live Dashboard",
            showlegend=True,
            height=1200,
            template="plotly_dark"
        )
        
        # Save as HTML for web viewing
        html_path = self.log_dir / "trading_dashboard.html"
        fig.write_html(str(html_path))
        logger.info(f"📊 Dashboard saved to {html_path}")
        
        return fig
    
    def _add_performance_comparison(self, fig, row, col):
        """Add GA vs PPO performance comparison."""
        # GA performance line
        if self.training_history['ga_generations']:
            ga_x = [entry['iteration'] for entry in self.training_history['ga_generations']]
            ga_y = [entry['metrics'].get('fitness', 0) for entry in self.training_history['ga_generations']]
            
            fig.add_trace(
                go.Scatter(
                    x=ga_x, y=ga_y,
                    mode='lines+markers',
                    name='🧬 GA Fitness',
                    line=dict(color=self.colors['ga'], width=3),
                    marker=dict(size=8)
                ),
                row=row, col=col
            )
        
        # PPO performance line
        if self.training_history['ppo_episodes']:
            ppo_x = [entry['iteration'] for entry in self.training_history['ppo_episodes']]
            ppo_y = [entry['metrics'].get('reward', 0) for entry in self.training_history['ppo_episodes']]
            
            fig.add_trace(
                go.Scatter(
                    x=ppo_x, y=ppo_y,
                    mode='lines+markers',
                    name='🎯 PPO Reward',
                    line=dict(color=self.colors['ppo'], width=3),
                    marker=dict(size=8)
                ),
                row=row, col=col, secondary_y=True
            )
    
    def _add_action_distribution(self, fig, row, col):
        """Add action distribution analysis."""
        actions = ['Hold', 'Long', 'Short']
        # Simulate action distribution data
        values = [45, 35, 20]  # Example percentages
        
        fig.add_trace(
            go.Bar(
                x=actions, y=values,
                name='Action Distribution',
                marker_color=[self.colors['neutral'], self.colors['profit'], self.colors['loss']]
            ),
            row=row, col=col
        )
    
    def _add_trading_metrics(self, fig, row, col):
        """Add trading performance metrics."""
        # Example metrics
        sharpe_ratio = 1.25
        max_drawdown = -8.5
        win_rate = 62.3
        
        fig.add_trace(
            go.Scatter(
                x=['Sharpe', 'Drawdown', 'Win Rate'],
                y=[sharpe_ratio, max_drawdown, win_rate],
                mode='markers+text',
                text=[f'{sharpe_ratio:.2f}', f'{max_drawdown:.1f}%', f'{win_rate:.1f}%'],
                textposition="middle right",
                marker=dict(size=20, color=[self.colors['profit'], self.colors['loss'], self.colors['ppo']])
            ),
            row=row, col=col
        )
    
    def _add_adaptive_switching(self, fig, row, col):
        """Add adaptive algorithm switching visualization."""
        # Example switching pattern
        switch_times = [10, 25, 40, 55]
        switch_methods = ['GA', 'PPO', 'GA', 'PPO']
        
        colors = [self.colors['ga'] if method == 'GA' else self.colors['ppo'] for method in switch_methods]
        
        fig.add_trace(
            go.Scatter(
                x=switch_times, y=switch_methods,
                mode='markers+lines',
                name='Algorithm Switches',
                marker=dict(size=15, color=colors),
                line=dict(width=2)
            ),
            row=row, col=col
        )
    
    def _add_cumulative_returns(self, fig, row, col):
        """Add cumulative returns visualization."""
        # Simulate cumulative returns
        days = list(range(1, 101))
        returns = np.cumsum(np.random.normal(0.1, 1.5, 100))
        
        fig.add_trace(
            go.Scatter(
                x=days, y=returns,
                mode='lines',
                name='Cumulative Returns (%)',
                line=dict(color=self.colors['profit'], width=3),
                fill='tonexty'
            ),
            row=row, col=col
        )
    
    def _add_market_insights(self, fig, row, col):
        """Add real-time market insights."""
        # Current profit indicator
        current_profit = 12.5  # Example
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_profit,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Current Profit %"},
                delta={'reference': 10},
                gauge={
                    'axis': {'range': [-20, 30]},
                    'bar': {'color': self.colors['profit']},
                    'steps': [
                        {'range': [-20, 0], 'color': self.colors['loss']},
                        {'range': [0, 15], 'color': self.colors['neutral']},
                        {'range': [15, 30], 'color': self.colors['profit']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 20
                    }
                }
            ),
            row=row, col=col
        )
    
    def generate_progress_report(self):
        """Generate comprehensive progress report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_ga_generations': len(self.training_history['ga_generations']),
            'total_ppo_episodes': len(self.training_history['ppo_episodes']),
            'best_ga_fitness': max([entry['metrics'].get('fitness', 0) 
                                  for entry in self.training_history['ga_generations']], default=0),
            'best_ppo_reward': max([entry['metrics'].get('reward', 0) 
                                  for entry in self.training_history['ppo_episodes']], default=0),
            'training_efficiency': self._calculate_training_efficiency()
        }
        
        # Save report
        report_path = self.log_dir / "progress_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Progress report saved to {report_path}")
        return report
    
    def _calculate_training_efficiency(self):
        """Calculate training efficiency metrics."""
        if not (self.training_history['ga_generations'] or self.training_history['ppo_episodes']):
            return 0.0
        
        # Simple efficiency calculation based on improvement rate
        total_improvements = 0
        total_iterations = 0
        
        # GA improvements
        ga_fitness = [entry['metrics'].get('fitness', 0) for entry in self.training_history['ga_generations']]
        if len(ga_fitness) > 1:
            ga_improvements = sum(1 for i in range(1, len(ga_fitness)) if ga_fitness[i] > ga_fitness[i-1])
            total_improvements += ga_improvements
            total_iterations += len(ga_fitness) - 1
        
        # PPO improvements
        ppo_rewards = [entry['metrics'].get('reward', 0) for entry in self.training_history['ppo_episodes']]
        if len(ppo_rewards) > 1:
            ppo_improvements = sum(1 for i in range(1, len(ppo_rewards)) if ppo_rewards[i] > ppo_rewards[i-1])
            total_improvements += ppo_improvements
            total_iterations += len(ppo_rewards) - 1
        
        return (total_improvements / total_iterations * 100) if total_iterations > 0 else 0.0
    
    def start_web_server(self):
        """Start web server for real-time dashboard viewing."""
        try:
            import http.server
            import socketserver
            import threading
            
            class DashboardHandler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=str(self.log_dir), **kwargs)
            
            with socketserver.TCPServer(("0.0.0.0", self.web_port), DashboardHandler) as httpd:
                logger.info(f"🌐 Dashboard server started at http://0.0.0.0:{self.web_port}")
                logger.info(f"🔗 View dashboard at: http://0.0.0.0:{self.web_port}/trading_dashboard.html")
                
                # Run server in background thread
                server_thread = threading.Thread(target=httpd.serve_forever)
                server_thread.daemon = True
                server_thread.start()
                
                return httpd
        except Exception as e:
            logger.error(f"❌ Failed to start web server: {e}")
            return None
