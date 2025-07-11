
#!/usr/bin/env python
"""
Revolutionary Hybrid GA-PPO Trainer
===================================

Innovative approach that combines:
1. GA for exploration and robust policy discovery
2. PPO for fine-tuning and gradient-based optimization
3. Dynamic knowledge transfer between algorithms
4. Adaptive algorithm switching based on market conditions
5. Multi-objective optimization with trading-specific metrics
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Dict, List, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ga_policy_evolution import run_ga_evolution, PolicyNetwork
from policy_gradient_methods import PPOTrainer, ActorCriticNet
from utils import evaluate_agent_distributed, compute_performance_metrics
from trading_dashboard import TradingDashboard

logger = logging.getLogger(__name__)


class KnowledgeTransferModule:
    """Facilitates knowledge transfer between GA and PPO."""
    
    def __init__(self, input_dim: int, action_dim: int, device: str = "cpu"):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.device = device
        self.knowledge_bank = []
        
    def extract_ga_knowledge(self, ga_population: List[np.ndarray], 
                           fitness_scores: List[float]) -> Dict:
        """Extract knowledge from GA population."""
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[-10:]  # Top 10
        elite_policies = [ga_population[i] for i in sorted_indices]
        
        # Analyze policy diversity
        policy_matrix = np.vstack(elite_policies)
        diversity_score = np.std(policy_matrix, axis=0).mean()
        
        # Extract behavioral patterns
        return {
            'elite_policies': elite_policies,
            'diversity_score': diversity_score,
            'fitness_range': max(fitness_scores) - min(fitness_scores),
            'convergence_rate': self._calculate_convergence_rate(fitness_scores)
        }
        
    def transfer_to_ppo(self, ppo_model: ActorCriticNet, ga_knowledge: Dict) -> bool:
        """Transfer GA knowledge to PPO model."""
        try:
            elite_policies = ga_knowledge['elite_policies']
            if not elite_policies:
                return False
                
            # Average elite policies for initialization
            avg_policy = np.mean(elite_policies, axis=0)
            
            # Create temporary GA network to extract structured knowledge
            temp_ga = PolicyNetwork(self.input_dim, 64, self.action_dim, self.device)
            temp_ga.set_params(avg_policy)
            
            # Transfer base network weights (shared architecture)
            with torch.no_grad():
                # Copy base network
                ppo_model.base.load_state_dict(temp_ga.base.state_dict())
                
                # Initialize policy head with GA knowledge but keep some randomness
                ga_policy_weights = temp_ga.policy_head.state_dict()
                ppo_policy_weights = ppo_model.policy_head.state_dict()
                
                for key in ga_policy_weights:
                    # Blend GA knowledge with current PPO weights
                    blend_ratio = 0.7  # 70% GA, 30% current
                    ppo_policy_weights[key] = (
                        blend_ratio * ga_policy_weights[key] + 
                        (1 - blend_ratio) * ppo_policy_weights[key]
                    )
                
                ppo_model.policy_head.load_state_dict(ppo_policy_weights)
                
            logger.info("Successfully transferred GA knowledge to PPO")
            return True
            
        except Exception as e:
            logger.error(f"Knowledge transfer failed: {e}")
            return False
            
    def transfer_to_ga(self, ga_population: List[np.ndarray], 
                      ppo_model: ActorCriticNet, injection_rate: float = 0.2) -> List[np.ndarray]:
        """Inject PPO knowledge into GA population."""
        try:
            # Extract PPO policy parameters
            ppo_params = []
            with torch.no_grad():
                for param in ppo_model.base.parameters():
                    ppo_params.append(param.cpu().numpy().flatten())
                for param in ppo_model.policy_head.parameters():
                    ppo_params.append(param.cpu().numpy().flatten())
            
            ppo_policy_vector = np.concatenate(ppo_params)
            
            # Inject into population
            num_inject = int(len(ga_population) * injection_rate)
            inject_indices = np.random.choice(len(ga_population), num_inject, replace=False)
            
            for idx in inject_indices:
                # Add noise to PPO policy to maintain diversity
                noise_scale = 0.01
                noisy_policy = ppo_policy_vector + np.random.normal(0, noise_scale, ppo_policy_vector.shape)
                ga_population[idx] = noisy_policy
                
            logger.info(f"Injected PPO knowledge into {num_inject} GA individuals")
            return ga_population
            
        except Exception as e:
            logger.error(f"PPO to GA transfer failed: {e}")
            return ga_population
            
    def _calculate_convergence_rate(self, fitness_history: List[float]) -> float:
        """Calculate how quickly the population is converging."""
        if len(fitness_history) < 10:
            return 0.0
        recent_improvement = fitness_history[-1] - fitness_history[-10]
        return max(0.0, recent_improvement)


class MarketConditionAnalyzer:
    """Analyze market conditions to guide algorithm selection."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)
        self.volume_history = deque(maxlen=window_size)
        self.volatility_history = deque(maxlen=window_size)
        
    def update(self, price: float, volume: float):
        """Update market data."""
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Calculate volatility
        if len(self.price_history) > 1:
            returns = np.diff(list(self.price_history)) / np.array(list(self.price_history)[:-1])
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
            self.volatility_history.append(volatility)
            
    def get_market_regime(self) -> str:
        """Determine current market regime."""
        if len(self.price_history) < 20:
            return "insufficient_data"
            
        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))
        
        # Trend analysis
        recent_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        trend_strength = abs(recent_trend) / np.mean(prices)
        
        # Volatility analysis
        current_vol = self.volatility_history[-1] if self.volatility_history else 0.0
        avg_vol = np.mean(list(self.volatility_history)) if self.volatility_history else 0.0
        
        # Volume analysis
        recent_volume = np.mean(volumes[-10:])
        avg_volume = np.mean(volumes)
        
        # Determine regime
        if trend_strength > 0.005 and current_vol < avg_vol * 1.2:
            return "trending_stable"
        elif current_vol > avg_vol * 1.5:
            return "high_volatility"
        elif recent_volume < avg_volume * 0.8:
            return "low_liquidity"
        elif trend_strength < 0.002:
            return "sideways"
        else:
            return "normal"
            
    def recommend_algorithm(self, performance_history: Dict) -> str:
        """Recommend which algorithm to use based on conditions."""
        regime = self.get_market_regime()
        
        # Get recent performance
        ga_performance = performance_history.get('ga_recent', 0.0)
        ppo_performance = performance_history.get('ppo_recent', 0.0)
        
        # Algorithm selection logic
        if regime in ["high_volatility", "trending_stable"]:
            # GA better for exploration in volatile/trending markets
            return "ga" if ga_performance >= ppo_performance * 0.8 else "ppo"
        elif regime in ["low_liquidity", "sideways"]:
            # PPO better for exploitation in stable markets
            return "ppo" if ppo_performance >= ga_performance * 0.8 else "ga"
        else:
            # Use best performing algorithm
            return "ga" if ga_performance > ppo_performance else "ppo"


class HybridGAPPOTrainer:
    """Revolutionary hybrid trainer combining GA and PPO with intelligent switching."""
    
    def __init__(self, train_env, test_env, input_dim: int, action_dim: int,
                 device: str = "cpu", models_dir: str = "./models"):
        self.train_env = train_env
        self.test_env = test_env
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.device = device
        self.models_dir = models_dir
        
        # Initialize components
        self.knowledge_transfer = KnowledgeTransferModule(input_dim, action_dim, device)
        self.market_analyzer = MarketConditionAnalyzer()
        self.trading_dashboard = TradingDashboard()
        
        # Initialize models
        self.ga_model = PolicyNetwork(input_dim, 64, action_dim, device)
        self.ppo_trainer = PPOTrainer(
            train_env, input_dim, action_dim, device=device,
            model_save_path=os.path.join(models_dir, "hybrid_ppo.pth")
        )
        
        # Performance tracking
        self.performance_history = {
            'ga_scores': [],
            'ppo_scores': [],
            'hybrid_scores': [],
            'algorithm_switches': [],
            'market_regimes': []
        }
        
        # TensorBoard logging
        self.tb_writer = SummaryWriter("runs/hybrid_ga_ppo")
        
        # Current state
        self.current_algorithm = "ga"
        self.iteration = 0
        self.last_switch_iteration = 0
        self.switch_threshold = 5  # Minimum iterations between switches
        
    def train_hybrid(self, max_iterations: int = 50, 
                    ga_generations: int = 20, ppo_updates: int = 10) -> Dict:
        """Main hybrid training loop with adaptive switching."""
        
        logger.info("🚀 Starting Revolutionary Hybrid GA-PPO Training")
        
        best_performance = -float('inf')
        best_agent = None
        stagnation_count = 0
        
        with tqdm(range(max_iterations), desc="Hybrid Training") as pbar:
            for iteration in range(max_iterations):
                self.iteration = iteration
                
                # Update market conditions
                self._update_market_conditions()
                
                # Determine which algorithm to use
                algorithm_choice = self._select_algorithm()
                
                # Train with selected algorithm
                if algorithm_choice == "ga":
                    performance = self._train_ga_phase(ga_generations)
                    self.performance_history['ga_scores'].append(performance)
                    
                elif algorithm_choice == "ppo":
                    performance = self._train_ppo_phase(ppo_updates)
                    self.performance_history['ppo_scores'].append(performance)
                    
                else:  # hybrid
                    performance = self._train_hybrid_phase(ga_generations, ppo_updates)
                    
                self.performance_history['hybrid_scores'].append(performance)
                
                # Track best performance
                if performance > best_performance:
                    best_performance = performance
                    best_agent = self._get_current_best_agent()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    
                # Log progress
                self._log_iteration_progress(iteration, algorithm_choice, performance)
                
                # Update progress bar
                pbar.set_postfix({
                    'performance': f"{performance:.4f}",
                    'algorithm': algorithm_choice,
                    'best': f"{best_performance:.4f}",
                    'stagnation': stagnation_count
                })
                
                # Early stopping if stagnated
                if stagnation_count >= 10:
                    logger.info("Training stopped due to stagnation")
                    break
                    
        logger.info(f"✅ Hybrid training completed. Best performance: {best_performance:.4f}")
        return {
            'best_agent': best_agent,
            'best_performance': best_performance,
            'performance_history': self.performance_history
        }
        
    def _select_algorithm(self) -> str:
        """Select algorithm based on market conditions and performance."""
        # Prevent too frequent switching
        if self.iteration - self.last_switch_iteration < self.switch_threshold:
            return self.current_algorithm
            
        # Get market recommendation
        recent_performance = {
            'ga_recent': np.mean(self.performance_history['ga_scores'][-5:]) if self.performance_history['ga_scores'] else 0.0,
            'ppo_recent': np.mean(self.performance_history['ppo_scores'][-5:]) if self.performance_history['ppo_scores'] else 0.0
        }
        
        recommended = self.market_analyzer.recommend_algorithm(recent_performance)
        
        # Check if we should switch
        if recommended != self.current_algorithm:
            self.current_algorithm = recommended
            self.last_switch_iteration = self.iteration
            self.performance_history['algorithm_switches'].append({
                'iteration': self.iteration,
                'from': self.current_algorithm,
                'to': recommended,
                'reason': self.market_analyzer.get_market_regime()
            })
            
        return self.current_algorithm
        
    def _train_ga_phase(self, generations: int) -> float:
        """Train using GA with knowledge transfer."""
        logger.info(f"🧬 GA Phase: {generations} generations")
        
        # Run GA evolution
        ga_agent, ga_fitness, ga_population, fitness_scores = run_ga_evolution(
            self.train_env,
            population_size=40,
            generations=generations,
            device=self.device,
            model_save_path=os.path.join(self.models_dir, "hybrid_ga.pth")
        )
        
        # Extract knowledge for transfer
        if ga_population and fitness_scores:
            ga_knowledge = self.knowledge_transfer.extract_ga_knowledge(
                ga_population, fitness_scores
            )
            
            # Transfer to PPO if beneficial
            if ga_knowledge['diversity_score'] > 0.1:  # Good diversity
                self.knowledge_transfer.transfer_to_ppo(
                    self.ppo_trainer.model, ga_knowledge
                )
                
        self.ga_model = ga_agent
        return ga_fitness
        
    def _train_ppo_phase(self, updates: int) -> float:
        """Train using PPO with knowledge transfer."""
        logger.info(f"🎯 PPO Phase: {updates} updates")
        
        # Inject GA knowledge into PPO if available
        if hasattr(self, 'ga_model') and self.ga_model:
            # This is a placeholder - actual implementation would require
            # modifying PPO trainer to accept knowledge injection
            pass
            
        # Train PPO
        initial_performance = self._evaluate_current_performance()
        
        for _ in range(updates):
            self.ppo_trainer.train_step()
            
        final_performance = self._evaluate_current_performance()
        improvement = final_performance - initial_performance
        
        return final_performance
        
    def _train_hybrid_phase(self, ga_generations: int, ppo_updates: int) -> float:
        """Train using both algorithms with knowledge transfer."""
        logger.info("🔄 Hybrid Phase: GA + PPO with knowledge transfer")
        
        # First, GA phase
        ga_performance = self._train_ga_phase(ga_generations // 2)
        
        # Transfer GA knowledge to PPO
        if hasattr(self, 'ga_model'):
            temp_population = [self.ga_model.get_params()]
            temp_fitness = [ga_performance]
            ga_knowledge = self.knowledge_transfer.extract_ga_knowledge(
                temp_population, temp_fitness
            )
            self.knowledge_transfer.transfer_to_ppo(self.ppo_trainer.model, ga_knowledge)
            
        # Then, PPO phase
        ppo_performance = self._train_ppo_phase(ppo_updates)
        
        # Transfer PPO knowledge back to GA population (for next iteration)
        # This would be used in the next GA phase
        
        return max(ga_performance, ppo_performance)
        
    def _update_market_conditions(self):
        """Update market analyzer with recent data."""
        # Get recent market data from environment
        if hasattr(self.train_env, 'states') and self.train_env.states:
            recent_states = self.train_env.states[-10:]  # Last 10 states
            for state in recent_states:
                self.market_analyzer.update(state.close_price, state.volume or 1000)
                
        # Record current regime
        current_regime = self.market_analyzer.get_market_regime()
        self.performance_history['market_regimes'].append(current_regime)
        
    def _evaluate_current_performance(self) -> float:
        """Evaluate current best agent performance."""
        current_agent = self._get_current_best_agent()
        if current_agent is None:
            return 0.0
            
        try:
            profits, times = evaluate_agent_distributed(self.test_env, current_agent, 0)
            if len(profits) > 1:
                cagr, sharpe, mdd = compute_performance_metrics(profits, times)
                # Composite score favoring risk-adjusted returns
                return cagr * 0.4 + sharpe * 0.4 - mdd * 0.2
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.0
            
    def _get_current_best_agent(self):
        """Get the current best performing agent."""
        if self.current_algorithm == "ga" and hasattr(self, 'ga_model'):
            return self.ga_model
        elif self.current_algorithm == "ppo":
            return self.ppo_trainer.model
        else:
            # Return best of both
            ga_performance = self._evaluate_agent(self.ga_model) if hasattr(self, 'ga_model') else -float('inf')
            ppo_performance = self._evaluate_agent(self.ppo_trainer.model)
            
            if ga_performance > ppo_performance:
                return self.ga_model
            else:
                return self.ppo_trainer.model
                
    def _evaluate_agent(self, agent) -> float:
        """Quick evaluation of an agent."""
        try:
            profits, times = evaluate_agent_distributed(self.test_env, agent, 0)
            if len(profits) > 1:
                cagr, sharpe, mdd = compute_performance_metrics(profits, times)
                return cagr * 0.4 + sharpe * 0.4 - mdd * 0.2
            return 0.0
        except:
            return 0.0
            
    def _log_iteration_progress(self, iteration: int, algorithm: str, performance: float):
        """Log detailed progress to TensorBoard."""
        # Basic metrics
        self.tb_writer.add_scalar("Hybrid/Performance", performance, iteration)
        self.tb_writer.add_scalar("Hybrid/Algorithm", 1 if algorithm == "ga" else 0, iteration)
        
        # Market conditions
        regime = self.market_analyzer.get_market_regime()
        regime_encoding = {
            "trending_stable": 0, "high_volatility": 1, "low_liquidity": 2,
            "sideways": 3, "normal": 4, "insufficient_data": 5
        }
        self.tb_writer.add_scalar("Market/Regime", regime_encoding.get(regime, 5), iteration)
        
        # Performance comparison
        if self.performance_history['ga_scores']:
            self.tb_writer.add_scalar("Performance/GA_Recent", 
                                    np.mean(self.performance_history['ga_scores'][-5:]), iteration)
        if self.performance_history['ppo_scores']:
            self.tb_writer.add_scalar("Performance/PPO_Recent",
                                    np.mean(self.performance_history['ppo_scores'][-5:]), iteration)
                                    
        # Create performance visualization
        if iteration % 10 == 0:
            self._create_performance_visualization(iteration)
            
    def _create_performance_visualization(self, iteration: int):
        """Create comprehensive performance visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance over time
        ax1.plot(self.performance_history['hybrid_scores'], 'b-', label='Hybrid', linewidth=2)
        if self.performance_history['ga_scores']:
            ax1.plot(self.performance_history['ga_scores'], 'r--', label='GA', alpha=0.7)
        if self.performance_history['ppo_scores']:
            ax1.plot(self.performance_history['ppo_scores'], 'g--', label='PPO', alpha=0.7)
        ax1.set_title("Performance Over Time")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Performance Score")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Algorithm usage
        algorithms = [switch.get('to', 'unknown') for switch in self.performance_history['algorithm_switches']]
        if algorithms:
            unique_algs, counts = np.unique(algorithms, return_counts=True)
            ax2.pie(counts, labels=unique_algs, autopct='%1.1f%%')
            ax2.set_title("Algorithm Usage Distribution")
        
        # Market regimes
        regimes = self.performance_history['market_regimes']
        if regimes:
            unique_regimes, regime_counts = np.unique(regimes, return_counts=True)
            ax3.bar(unique_regimes, regime_counts)
            ax3.set_title("Market Regime Distribution")
            ax3.tick_params(axis='x', rotation=45)
        
        # Performance vs market regime
        if regimes and self.performance_history['hybrid_scores']:
            regime_performance = {}
            for i, regime in enumerate(regimes):
                if i < len(self.performance_history['hybrid_scores']):
                    if regime not in regime_performance:
                        regime_performance[regime] = []
                    regime_performance[regime].append(self.performance_history['hybrid_scores'][i])
            
            if regime_performance:
                regimes_list = list(regime_performance.keys())
                avg_performance = [np.mean(regime_performance[r]) for r in regimes_list]
                ax4.bar(regimes_list, avg_performance)
                ax4.set_title("Average Performance by Market Regime")
                ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.tb_writer.add_figure("Hybrid/Comprehensive_Analysis", fig, iteration)
        plt.close(fig)
        
    def save_best_model(self, filepath: str):
        """Save the best performing model."""
        best_agent = self._get_current_best_agent()
        if best_agent and hasattr(best_agent, 'save_model'):
            best_agent.save_model(filepath)
            logger.info(f"Best hybrid model saved to {filepath}")
