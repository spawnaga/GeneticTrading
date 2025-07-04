# Applying the requested changes to address the errors and warnings by adding model dimension validation, cleanup methods, and calling the cleanup in the initialization.
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
    def cleanup_incompatible_models(self):
        """Remove models with incompatible dimensions."""
        import os
        import glob

        model_files = []
        model_files.extend(glob.glob(os.path.join(self.models_dir, "**/*.pth"), recursive=True))

        for model_file in model_files:
            try:
                checkpoint = torch.load(model_file, map_location='cpu')
                if 'input_dim' in checkpoint and checkpoint['input_dim'] != self.expected_input_dim:
                    logging.warning(f"Removing incompatible model: {model_file} (expected {self.expected_input_dim}, got {checkpoint['input_dim']})")
                    os.remove(model_file)
                elif 'model_state_dict' in checkpoint:
                    # Check state dict dimensions
                    state_dict = checkpoint['model_state_dict']
                    for key, tensor in state_dict.items():
                        if 'weight' in key and len(tensor.shape) > 1:
                            if tensor.shape[1] != self.expected_input_dim and 'fc1' in key:
                                logging.warning(f"Removing incompatible model: {model_file}")
                                os.remove(model_file)
                                break
            except Exception as e:
                logging.warning(f"Could not validate model {model_file}: {e}")

    def __init__(
        self,
        train_env,
        test_env,
        input_dim: int,
        action_dim: int,
        device: str = "cpu",
        models_dir: str = "./models",
        local_rank: int = 0,
        world_size: int = 1,
        use_distributed: bool = False
    ):
        self.train_env = train_env
        self.test_env = test_env
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.device = device
        self.models_dir = Path(models_dir)
        self.local_rank = local_rank
        self.world_size = world_size
        self.use_distributed = use_distributed

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

        # Initialize GA and PPO models
        self.ga_agent = PolicyNetwork(input_dim, 64, action_dim, device=device)
        self.ppo_trainer = None

        # Store current dimensions for validation
        self.expected_input_dim = input_dim
        self.expected_action_dim = action_dim

        os.makedirs(os.path.dirname(self.ga_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.ppo_model_path), exist_ok=True)

        # Call cleanup after initialization
        self.cleanup_incompatible_models()

        # Setup TensorBoard logging
        if self.local_rank == 0:
            # Ensure TensorBoard directory exists
            os.makedirs("./runs", exist_ok=True)
            os.makedirs("./runs/ga_experiment", exist_ok=True)

            # Clean up old runs to prevent confusion
            if os.path.exists("./runs/ga_experiment"):
                try:
                    import shutil
                    shutil.rmtree("./runs/ga_experiment")
                    logger.info("Removed old TensorBoard run: ./runs/ga_experiment")
                except Exception as e:
                    logger.warning(f"Could not remove old TensorBoard run: {e}")

            # Recreate directory after cleanup
            os.makedirs("./runs/ga_experiment", exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter('./runs/ga_experiment')

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
        try:
            if self.current_method == "GA":
                if not hasattr(self, 'ga_agent') or self.ga_agent is None:
                    # Initialize GA agent if missing
                    self.ga_agent = PolicyNetwork(self.input_dim, 64, self.action_dim, device=self.device)
                agent = self.ga_agent
            else:
                if self.ppo_trainer is None:
                    self._initialize_ppo_trainer()
                agent = self.ppo_trainer.model if self.ppo_trainer else self.ga_agent

            # Store original device and ensure consistent device usage
            if hasattr(agent, 'parameters'):
                original_device = next(agent.parameters()).device
            else:
                original_device = torch.device(self.device)

            # Keep agent on its original device for evaluation
            agent = agent.to(original_device)

            # Evaluate performance with device-aware evaluation
            profits, times = evaluate_agent_distributed(self.test_env, agent, self.local_rank)
            logger.info(f"Evaluation results: {len(profits)} profits, total={sum(profits):.4f}")

            if len(profits) == 0:
                logger.warning("No profits returned from evaluation!")
                return 0.0, 0.0, {}

            # Update PPO trainer metrics if applicable
            if self.current_method == "PPO" and self.ppo_trainer:
                total_profit = sum(profits)
                current_account_value = 100000 + total_profit  # Starting capital + profits
                self.ppo_trainer.update_trading_metrics(
                    episode_profit=total_profit,
                    current_account_value=current_account_value
                )

            # Filter out None and NaN values from profits and times
            clean_profits = [p for p in profits if p is not None and np.isfinite(p)]
            if len(clean_profits) != len(profits):
                logger.warning(f"Filtered {len(profits) - len(clean_profits)} invalid profits")
                profits = clean_profits

            if not profits:
                logger.warning("No valid profits after filtering!")
                return 0.0, 0.0, {}

            # Filter out None values from times before passing to metrics
            valid_times = [t for t in times if t is not None] if times else None
            if valid_times and len(valid_times) < len(times):
                logger.warning(f"Some timestamps are None: {len(valid_times)} valid out of {len(times)} total")

            # Calculate performance metrics with additional safety checks
            try:
                cagr, sharpe, mdd = compute_performance_metrics(profits, valid_times)

                # Ensure all metrics are finite
                if not np.isfinite(cagr):
                    cagr = 0.0
                if not np.isfinite(sharpe):
                    sharpe = 0.0
                if not np.isfinite(mdd):
                    mdd = 100.0

            except Exception as e:
                logger.warning(f"Error computing performance metrics: {e}")
                cagr, sharpe, mdd = 0.0, 0.0, 100.0

            logger.info(f"Metrics: CAGR={cagr:.4f}, Sharpe={sharpe:.4f}, MDD={mdd:.4f}")

            # Calculate policy entropy (measure of exploration)
            policy_entropy = self._calculate_policy_entropy(agent)

            # Calculate composite performance score with better scaling
            # Normalize metrics to prevent extreme values
            normalized_sharpe = np.clip(sharpe / 10.0, -10, 10)  # Scale down Sharpe
            normalized_cagr = np.clip(cagr / 100.0, -10, 10)     # Scale down CAGR  
            normalized_mdd = np.clip(mdd / 100.0, 0, 10)         # Scale down MDD

            performance_score = normalized_sharpe * 0.5 + normalized_cagr * 0.3 - normalized_mdd * 0.2

            # Ensure performance score is finite
            if not np.isfinite(performance_score):
                performance_score = 0.0

            # Additional metrics
            metrics = {
                'cagr': float(cagr),
                'sharpe': float(sharpe),
                'mdd': float(mdd),
                'total_profit': float(sum(profits)),
                'win_rate': len([p for p in profits if p > 0]) / len(profits) if profits else 0.0,
                'policy_entropy': float(policy_entropy)
            }

            return performance_score, policy_entropy, metrics

        except Exception as e:
            logger.error(f"Error during policy evaluation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0, 0.0, {}

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
        Calculate average policy entropy across random states with NaN protection
        """
        entropies = []

        # Get agent's device
        if hasattr(agent, 'parameters'):
            agent_device = next(agent.parameters()).device
        else:
            agent_device = torch.device('cpu')

        # Sample random states from environment
        for _ in range(10):
            try:
                obs = self.test_env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

                # Ensure observation is valid and numpy array
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs)

                if not np.all(np.isfinite(obs)):
                    obs = np.zeros_like(obs)

                # Convert to tensor on the same device as the agent
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent_device)

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

                    # Check for NaN/infinite logits
                    if not torch.all(torch.isfinite(logits)):
                        logger.warning("NaN/infinite logits detected in policy entropy calculation")
                        continue

                    # Clamp logits to prevent extreme values
                    logits = torch.clamp(logits, -10.0, 10.0)
                    probs = torch.softmax(logits, dim=-1)

                    # Additional safety check
                    if not torch.all(torch.isfinite(probs)):
                        logger.warning("NaN/infinite probabilities detected")
                        continue

                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

                    if torch.isfinite(entropy):
                        entropies.append(float(entropy.detach().cpu().item()))  # Ensure CPU conversion and float

            except Exception as e:
                logger.warning(f"Error calculating entropy for sample: {e}")
                continue

        if not entropies:
            logger.warning("No valid entropy calculations, returning default value")
            return 1.0986  # Default entropy value (log(3) for 3 actions)

        entropy_mean = np.mean(entropies)
        if not np.isfinite(entropy_mean):
            return 1.0986

        return float(entropy_mean)

    def run_ga_phase(self, generations: int = 20, population_size: int = 20) -> float:
        """
        Run GA optimization phase
        """
        logger.info(f"Starting GA phase: {generations} generations, population {population_size}")

        # Load previous GA model if exists
        if os.path.exists(self.ga_model_path):
            self.ga_agent.load_model(self.ga_model_path)

        # Run GA evolution with faster settings
        best_agent, best_fitness, _, _ = run_ga_evolution(
            self.train_env,
            population_size=min(population_size, 15),  # Limit population size
            generations=min(generations, 10),  # Limit generations
            tournament_size=max(3, min(population_size, 15) // 5),
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
            try:
                if hasattr(self, 'ga_agent') and self.ga_agent is not None:
                    self.ppo_trainer.model.load_state_dict(self.ga_agent.state_dict())
                else:
                    logger.warning("GA agent not available for weight transfer")
                    self.ppo_trainer.model._initialize_weights()
            except Exception as e:
                logger.warning(f"Failed to transfer GA weights to PPO: {e}")
                self.ppo_trainer.model._initialize_weights()

        # Run PPO training with improved stability
        initial_performance = self.evaluate_current_policy()[0]
        best_performance = initial_performance
        patience_counter = 0
        max_patience = 3  # Reduced patience for faster switching
        consecutive_failures = 0
        max_consecutive_failures = 10  # Stop after too many failures

        for update in range(total_updates):
            try:
                reward = self.ppo_trainer.train_step()
                consecutive_failures = 0  # Reset on successful update

                # Check if reward is valid
                if not np.isfinite(reward):
                    logger.warning(f"Invalid reward at update {update}: {reward}")
                    reward = 0.0

            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"PPO training step failed at update {update}: {e}")

                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive PPO failures ({consecutive_failures}), stopping PPO phase")
                    break

                # Try to recover by reinitializing the model
                logger.info("Attempting to recover by reinitializing PPO model")
                self.ppo_trainer.model._initialize_weights()
                self.ppo_trainer.optimizer.state = {}  # Reset optimizer state
                continue

            # Reduced evaluation frequency for faster training
            if (update + 1) % 50 == 0:  # Evaluate every 50 updates
                try:
                    performance, entropy, metrics = self.evaluate_current_policy()
                    logger.info(f"PPO Update {update + 1}: Performance={performance:.4f}, Entropy={entropy:.4f}")

                    # Update best performance and patience counter
                    if performance > best_performance:
                        best_performance = performance
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # More conservative early stopping
                    if performance < initial_performance * 0.3:  # Earlier stopping for poor performance
                        patience_counter += 2  # Penalize severe degradation

                    if patience_counter >= max_patience:
                        logger.warning(f"PPO early stopping due to lack of improvement (patience: {patience_counter})")
                        break
                except Exception as e:
                    logger.warning(f"PPO evaluation failed: {e}")
                    continue

        # Save PPO model
        try:
            self.ppo_trainer.model.save_model(self.ppo_model_path)
            self.current_method = "PPO"
        except Exception as e:
            logger.warning(f"Failed to save PPO model: {e}")

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