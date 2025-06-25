# Applying the change to ensure TensorBoard directories exist before creating writers.
import os
import logging
import numpy as np
import torch
from collections import deque
from typing import Dict, List, Optional, Tuple
import datetime
from pathlib import Path

from ga_policy_evolution import run_ga_evolution, PolicyNetwork
from policy_gradient_methods import PPOTrainer, ActorCriticNet
from utils import evaluate_agent_distributed, compute_performance_metrics

logger = logging.getLogger(__name__)


class AdaptiveTrainer:
    """
    Intelligent trainer that switches between GA and PPO based on performance conditions
    """

    def __init__(
        self,
        train_env,
        test_env,
        input_dim: int,
        action_dim: int,
        device: str = "cpu",
        models_dir: str = "./models",
        local_rank: int = 0
    ):
        self.train_env = train_env
        self.test_env = test_env
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.device = device
        self.models_dir = Path(models_dir)
        self.local_rank = local_rank

        # Ensure TensorBoard directories exist with proper permissions
        import os
        try:
            os.makedirs("./runs/ga_experiment", exist_ok=True)
            os.makedirs(f"./runs/ppo_rank_{local_rank}", exist_ok=True)
            # Create nested structure for better organization
            os.makedirs("./runs/ga_experiment/logs", exist_ok=True)
            os.makedirs(f"./runs/ppo_rank_{local_rank}/logs", exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create TensorBoard directories: {e}")
            # Use temp directory as fallback
            import tempfile
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Using temporary directory for TensorBoard: {temp_dir}")

        # Performance tracking
        self.performance_history = deque(maxlen=20)  # Last 20 evaluations
        self.stagnation_count = 0
        self.consecutive_poor_performance = 0
        self.best_performance = -float('inf')
        self.current_method = None

        # Thresholds for switching
        self.stagnation_threshold = 5  # Switch to GA if no improvement for N evaluations
        self.poor_performance_threshold = 3  # Switch to GA if poor performance for N evaluations
        self.exploitation_threshold = 0.8  # Switch to PPO if performing well
        self.diversity_threshold = 0.1  # Switch to GA if policy becomes too deterministic

        # Model paths
        self.ga_model_path = os.path.join(models_dir, "ga_models", "adaptive_ga_model.pth")
        self.ppo_model_path = os.path.join(models_dir, "ppo_models", "adaptive_ppo_model.pth")

        # Initialize models
        self.ga_agent = PolicyNetwork(input_dim, 64, action_dim, device=device)
        self.ppo_trainer = None

        os.makedirs(os.path.dirname(self.ga_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.ppo_model_path), exist_ok=True)

    def should_switch_to_ga(self, current_performance: float, policy_entropy: float) -> bool:
        """
        Determine if we should switch to GA based on current conditions
        """
        reasons = []

        # 1. Performance stagnation
        if self.stagnation_count >= self.stagnation_threshold:
            reasons.append("performance_stagnation")

        # 2. Consecutive poor performance
        if self.consecutive_poor_performance >= self.poor_performance_threshold:
            reasons.append("poor_performance")

        # 3. Low policy diversity (too deterministic)
        if policy_entropy < self.diversity_threshold:
            reasons.append("low_diversity")

        # 4. Market regime change detection (volatility spike)
        if len(self.performance_history) >= 5:
            recent_volatility = np.std(list(self.performance_history)[-5:])
            historical_volatility = np.std(list(self.performance_history))
            if recent_volatility > historical_volatility * 2:
                reasons.append("market_regime_change")

        # 5. Exploration phase (early training or after major changes)
        if len(self.performance_history) < 3:
            reasons.append("exploration_phase")

        if reasons:
            logger.info(f"Switching to GA due to: {', '.join(reasons)}")
            return True

        return False

    def should_switch_to_ppo(self, current_performance: float) -> bool:
        """
        Determine if we should switch to PPO for fine-tuning
        """
        reasons = []

        # 1. Good performance that can be refined
        if current_performance > self.best_performance * self.exploitation_threshold:
            reasons.append("good_performance_refinement")

        # 2. Stable performance (low volatility)
        if len(self.performance_history) >= 5:
            recent_std = np.std(list(self.performance_history)[-5:])
            if recent_std < np.std(list(self.performance_history)) * 0.5:
                reasons.append("stable_performance")

        # 3. GA has found a good solution that needs gradient-based refinement
        if self.current_method == "GA" and self.stagnation_count <= 2:
            reasons.append("ga_solution_refinement")

        if reasons:
            logger.info(f"Switching to PPO due to: {', '.join(reasons)}")
            return True

        return False

    def evaluate_current_policy(self) -> Tuple[float, float, Dict]:
        """
        Evaluate current policy and return performance metrics and policy entropy
        """
        if self.current_method == "GA":
            agent = self.ga_agent
        else:
            agent = self.ppo_trainer.model if self.ppo_trainer else self.ga_agent

        # Evaluate performance
        profits, times = evaluate_agent_distributed(self.test_env, agent, self.local_rank)
        logger.info(f"Evaluation results: {len(profits)} profits, total={sum(profits):.4f}")

        if len(profits) == 0:
            logger.warning("No profits returned from evaluation!")
            return 0.0, 0.0, {}

        cagr, sharpe, mdd = compute_performance_metrics(profits, times)
        logger.info(f"Metrics: CAGR={cagr:.4f}, Sharpe={sharpe:.4f}, MDD={mdd:.4f}")

        # Calculate composite performance score with better scaling
        # Normalize metrics to prevent extreme values
        normalized_sharpe = np.clip(sharpe / 10.0, -10, 10)  # Scale down Sharpe
        normalized_cagr = np.clip(cagr / 100.0, -10, 10)     # Scale down CAGR  
        normalized_mdd = np.clip(mdd / 100.0, 0, 10)         # Scale down MDD

        performance_score = normalized_sharpe * 0.5 + normalized_cagr * 0.3 - normalized_mdd * 0.2

        # Calculate policy entropy (measure of exploration)
        policy_entropy = self._calculate_policy_entropy(agent)

        # Additional metrics
        metrics = {
            'cagr': cagr,
            'sharpe': sharpe,
            'mdd': mdd,
            'total_profit': sum(profits),
            'win_rate': len([p for p in profits if p > 0]) / len(profits) if profits else 0,
            'policy_entropy': policy_entropy
        }

        return performance_score, policy_entropy, metrics

    def _calculate_policy_entropy(self, agent) -> float:
        """
        Calculate average policy entropy across random states
        """
        entropies = []

        # Sample random states from environment
        for _ in range(10):
            obs = self.test_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if hasattr(agent, 'forward'):
                    output = agent.forward(obs_tensor)
                    # Handle different agent types
                    if isinstance(output, tuple):
                        # PPO agent returns (logits, value)
                        logits, _ = output
                    else:
                        # GA agent returns only logits
                        logits = output
                else:
                    logits = agent(obs_tensor)

                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                entropies.append(entropy.item())

        return np.mean(entropies)

    def run_ga_phase(self, generations: int = 20, population_size: int = 20) -> float:
        """
        Run GA optimization phase
        """
        logger.info(f"Starting GA phase: {generations} generations, population {population_size}")

        # Load previous GA model if exists
        if os.path.exists(self.ga_model_path):
            self.ga_agent.load_model(self.ga_model_path)

        # Run GA evolution
        best_agent, best_fitness, _, _ = run_ga_evolution(
            self.train_env,
            population_size=population_size,
            generations=generations,
            tournament_size=max(3, population_size // 7),
            mutation_rate=0.4,
            mutation_scale=0.3,
            num_workers=1,
            device=self.device,
            model_save_path=self.ga_model_path,
            fitness_metric="comprehensive"
        )

        self.ga_agent = best_agent
        self.current_method = "GA"

        # Transfer GA weights to PPO model for potential future use
        if self.ppo_trainer is None:
            self._initialize_ppo_trainer()

        # Copy GA weights to PPO model
        self.ppo_trainer.model.load_state_dict(self.ga_agent.state_dict())

        logger.info(f"GA phase completed with fitness: {best_fitness:.4f}")
        return best_fitness

    def run_ppo_phase(self, total_updates: int = 100) -> float:
        """
        Run PPO optimization phase
        """
        logger.info(f"Starting PPO phase: {total_updates} updates")

        if self.ppo_trainer is None:
            self._initialize_ppo_trainer()

        # Load previous PPO model if exists, otherwise use GA weights
        if os.path.exists(self.ppo_model_path):
            self.ppo_trainer.model.load_model(self.ppo_model_path)
        else:
            # Transfer weights from GA agent
            self.ppo_trainer.model.load_state_dict(self.ga_agent.state_dict())

        # Run PPO training with improved stability
        initial_performance = self.evaluate_current_policy()[0]
        best_performance = initial_performance
        patience_counter = 0
        max_patience = 5  # Allow more patience for convergence

        for update in range(total_updates):
            try:
                reward = self.ppo_trainer.train_step()
            except Exception as e:
                logger.warning(f"PPO training step failed at update {update}: {e}")
                break

            # Periodic evaluation with improved early stopping
            if (update + 1) % 10 == 0:
                performance, entropy, metrics = self.evaluate_current_policy()
                logger.info(f"PPO Update {update + 1}: Performance={performance:.4f}, Entropy={entropy:.4f}")

                # Update best performance and patience counter
                if performance > best_performance:
                    best_performance = performance
                    patience_counter = 0
                else:
                    patience_counter += 1

                # More conservative early stopping
                if performance < initial_performance * 0.5:  # Allow more degradation
                    patience_counter += 2  # Penalize severe degradation

                if patience_counter >= max_patience:
                    logger.warning(f"PPO early stopping due to lack of improvement (patience: {patience_counter})")
                    break

        # Save PPO model
        self.ppo_trainer.model.save_model(self.ppo_model_path)
        self.current_method = "PPO"

        final_performance = self.evaluate_current_policy()[0]
        logger.info(f"PPO phase completed with performance: {final_performance:.4f}")
        return final_performance

    def _initialize_ppo_trainer(self):
        """
        Initialize PPO trainer if not already done
        """
        self.ppo_trainer = PPOTrainer(
            env=self.train_env,
            input_dim=self.input_dim,
            action_dim=self.action_dim,
            hidden_dim=64,
            lr=3e-4 * 0.5,  # Reduced learning rate for stability
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            update_epochs=4,
            rollout_steps=1024,
            batch_size=64,
            device=self.device,
            model_save_path=self.ppo_model_path,
            local_rank=self.local_rank,
            eval_interval=10
        )

    def adaptive_train(
        self,
        max_iterations: int = 50,
        evaluation_interval: int = 1
    ) -> Dict:
        """
        Main adaptive training loop that switches between GA and PPO
        """
        logger.info("Starting adaptive training")

        training_log = {
            'iterations': [],
            'methods': [],
            'performances': [],
            'metrics': [],
            'switch_reasons': []
        }

        # Start with GA for initial exploration
        self.current_method = "GA"

        for iteration in range(max_iterations):
            logger.info(f"\n=== Adaptive Training Iteration {iteration + 1}/{max_iterations} ===")

            # Evaluate current performance
            performance, entropy, metrics = self.evaluate_current_policy()

            # Update tracking
            self.performance_history.append(performance)

            # Check for improvement
            if performance > self.best_performance:
                self.best_performance = performance
                self.stagnation_count = 0
                self.consecutive_poor_performance = 0
            else:
                self.stagnation_count += 1
                if performance < np.mean(list(self.performance_history)[-5:]) * 0.9:
                    self.consecutive_poor_performance += 1
                else:
                    self.consecutive_poor_performance = 0

            logger.info(f"Current performance: {performance:.4f} (best: {self.best_performance:.4f})")
            logger.info(f"Stagnation: {self.stagnation_count}, Poor performance: {self.consecutive_poor_performance}")
            logger.info(f"Method: {self.current_method}, Entropy: {entropy:.4f}")

            # Decide on next method
            next_method = self.current_method
            switch_reason = "no_switch"

            if self.current_method == "PPO":
                if self.should_switch_to_ga(performance, entropy):
                    next_method = "GA"
                    switch_reason = "ppo_to_ga"
            else:  # current_method == "GA"
                if self.should_switch_to_ppo(performance):
                    next_method = "PPO"
                    switch_reason = "ga_to_ppo"

            # Log iteration results
            training_log['iterations'].append(iteration + 1)
            training_log['methods'].append(self.current_method)
            training_log['performances'].append(performance)
            training_log['metrics'].append(metrics.copy())
            training_log['switch_reasons'].append(switch_reason)

            # Execute training phase
            if next_method != self.current_method:
                logger.info(f"Switching from {self.current_method} to {next_method}")

            if next_method == "GA":
                # Determine GA parameters based on situation
                if self.consecutive_poor_performance > 0:
                    # More aggressive search for poor performance
                    generations = 30
                    population_size = 30
                else:
                    # Standard exploration
                    generations = 20
                    population_size = 20

                self.run_ga_phase(generations=generations, population_size=population_size)

            else:  # PPO
                # Determine PPO parameters based on situation
                if switch_reason == "ga_to_ppo":
                    # Longer refinement after GA
                    updates = 150
                else:
                    # Standard PPO updates
                    updates = 100

                self.run_ppo_phase(total_updates=updates)

            # Reset counters if we switched methods
            if next_method != self.current_method:
                self.stagnation_count = 0
                self.consecutive_poor_performance = 0

        logger.info("Adaptive training completed")
        return training_log

    def get_best_agent(self):
        """
        Return the best performing agent
        """
        # Evaluate both agents and return the better one
        ga_performance = 0
        ppo_performance = 0

        if os.path.exists(self.ga_model_path):
            self.ga_agent.load_model(self.ga_model_path)
            ga_performance = self.evaluate_current_policy()[0]

        if self.ppo_trainer and os.path.exists(self.ppo_model_path):
            self.ppo_trainer.model.load_model(self.ppo_model_path)
            original_method = self.current_method
            self.current_method = "PPO"
            ppo_performance = self.evaluate_current_policy()[0]
            self.current_method = original_method

        if ppo_performance > ga_performance:
            logger.info(f"Best agent: PPO (performance: {ppo_performance:.4f})")
            return self.ppo_trainer.model
        else:
            logger.info(f"Best agent: GA (performance: {ga_performance:.4f})")
            return self.ga_agent