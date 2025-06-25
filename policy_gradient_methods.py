# The progress bar in the collect_trajectories method and the train_step method are simplified to reduce verbosity.
# Corrected the _unpack_reset function call in the collect_trajectories method to handle the tuple return value from env.reset().
# policy_gradient_methods.py

"""
Proximal Policy Optimization (PPO) implementation using PyTorch and DistributedDataParallel.

This module defines:
  - ActorCriticNet: shared-base actor-critic network
  - PPOTrainer: collects trajectories, computes GAE, and updates policy/value heads
  - Comprehensive TensorBoard instrumentation with creative visualizations for deep insights
"""

import os
import datetime
import logging
import time
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent threading issues
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from utils import evaluate_agent_distributed, compute_performance_metrics, _unpack_reset, _unpack_step, cleanup_tensorboard_runs

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ActorCriticNet(nn.Module):
    """
    Shared-base actor-critic network.

    - base: two hidden layers with ReLU
    - policy_head: outputs logits for Discrete(action_dim)
    - value_head: outputs scalar state value
    """

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        # Shared base network with better initialization
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(self.device)
        # Policy logits head
        self.policy_head = nn.Linear(hidden_dim, action_dim).to(self.device)
        # State-value head
        self.value_head = nn.Linear(hidden_dim, 1).to(self.device)

        # Initialize weights with Xavier/Glorot initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights for stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # More conservative initialization to prevent NaN propagation
                nn.init.orthogonal_(module.weight, gain=0.01)  # Orthogonal initialization
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        Forward pass with NaN protection.
        Args:
            x: tensor of shape [batch, input_dim] or higher dimensions
        Returns:
            policy_logits: [batch, action_dim]
            value: [batch, 1]
        """
        x = x.to(self.device)

        # More aggressive input validation
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            # Replace NaN/inf with small random values instead of zeros
            mask = torch.isfinite(x)
            x = torch.where(mask, x, torch.randn_like(x) * 0.01)

        # Normalize input to prevent gradient explosion
        x = torch.clamp(x, -10.0, 10.0)

        # Add small amount of noise to prevent identical states
        x = x + torch.randn_like(x) * 1e-6

        if x.ndim > 2:
            x = x.view(x.size(0), -1)

        hidden = self.base(x)

        # More aggressive hidden layer protection
        if torch.any(torch.isnan(hidden)) or torch.any(torch.isinf(hidden)):
            hidden = torch.clamp(hidden, -5.0, 5.0)
            hidden = torch.where(torch.isfinite(hidden), hidden, torch.randn_like(hidden) * 0.01)

        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden)

        # Final robust output protection
        policy_logits = torch.clamp(policy_logits, -5.0, 5.0)
        policy_logits = torch.where(torch.isfinite(policy_logits), policy_logits, torch.randn_like(policy_logits) * 0.01)

        value = torch.clamp(value, -10.0, 10.0)
        value = torch.where(torch.isfinite(value), value, torch.zeros_like(value))

        return policy_logits, value

    def act(self, state):
        """
        Select action greedily (for evaluation).
        Args:
            state: numpy array or torch tensor, shape [input_dim]
        Returns:
            action: int
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        # Ensure proper shape and device
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        with torch.no_grad():
            logits, _ = self.forward(state)
            return torch.argmax(logits, dim=-1).item()

    def save_model(self, path: str, quiet: bool = False):
        """Save model state dict."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        if not quiet:
            logger.info(f"Saved model to {path} at {datetime.datetime.now()}")

    def load_model(self, path: str):
        """Load model state dict if exists, else leave random init."""
        if os.path.exists(path):
            try:
                saved_state = torch.load(path, map_location=self.device)

                # Check if dimensions match
                if 'base.0.weight' in saved_state:
                    saved_input_dim = saved_state['base.0.weight'].shape[1]
                    current_input_dim = self.base[0].weight.shape[1]

                    if saved_input_dim != current_input_dim:
                        logger.warning(f"Dimension mismatch: saved model expects {saved_input_dim}, current model has {current_input_dim}")
                        logger.info(f"Starting from scratch due to incompatible checkpoint at {path}")
                        return

                self.load_state_dict(saved_state)
                logger.info(f"Loaded model from {path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {path}: {e}")
                logger.info("Starting from scratch")
        else:
            logger.info(f"No model file at {path}, starting from scratch")


class PPOTrainer:
    """
    PPO trainer using clipped surrogate objective and GAE.

    Workflow:
      1. collect_trajectories() → rollouts
      2. compute_gae() → advantages, returns
      3. train_step() → multiple epochs of policy/value updates
      4. train() → loop over train_step() until total_timesteps
      5. Periodic evaluation and TensorBoard logging of all key metrics
    """

    def __init__(
        self,
        env,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        update_epochs: int = 10,
        rollout_steps: int = 2048,
        batch_size: int = 64,
        device: str = "cpu",
        model_save_path: str = "ppo_model.pth",
        local_rank: int = 0,
        eval_interval: int = 10   # run evaluation every `eval_interval` updates
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model_save_path = model_save_path
        self.local_rank = local_rank
        self.eval_interval = eval_interval

        # Build model (will wrap in DDP externally)
        self.model = ActorCriticNet(input_dim, hidden_dim, action_dim, device=device)
        self.model.load_model(self.model_save_path)

        # Optimizer & LR scheduler: decay once per update_step - reduced LR for stability
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr*0.3, eps=1e-5)  # Reduce LR by 70%
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.98)  # More gradual decay

        # Entropy coefficient for exploration: decay once per update_step
        self.entropy_coef = 0.05  # Higher initial exploration for better learning

        # Enhanced tracking for visualization
        self.action_distribution_history = deque(maxlen=100)
        self.reward_trends = deque(maxlen=1000)
        self.loss_trends = deque(maxlen=1000)
        self.gradient_norms = deque(maxlen=100)
        self.policy_divergence = deque(maxlen=100)
        self.value_accuracy = deque(maxlen=100)

        # Trading-specific metrics
        self.position_changes = deque(maxlen=1000)
        self.profit_per_trade = deque(maxlen=500)
        self.drawdown_periods = []

        # cleanup old TensorBoard runs before starting new one
        if local_rank == 0:
            cleanup_tensorboard_runs("./runs", keep_latest=1)

        # tensorboard
        self.tb_writer = SummaryWriter(
            log_dir=f"runs/ppo_rank_{local_rank}",
            flush_secs=1
        )
        self.global_step = 0

    def collect_trajectories(self):
        """
        Run `rollout_steps` steps in env to collect:
          observations, actions, log probabilities, values, rewards, dones
        Returns numpy arrays.
        """
        obs_buf, act_buf = [], []
        logp_buf, val_buf = [], []
        rew_buf, done_buf = [], []

        state = _unpack_reset(self.env.reset())

        # Add progress bar for trajectory collection
        trajectory_pbar = tqdm(range(self.rollout_steps), 
                              desc="Collecting trajectories", 
                              leave=False, 
                              disable=(self.local_rank != 0))

        for step in trajectory_pbar:
            # Ensure state is not empty and has proper shape
            if len(state) == 0:
                # Create a dummy observation if state is empty
                state = np.zeros(self.env.observation_space.shape, dtype=np.float32)

            # Validate state before creating tensor
            if not np.all(np.isfinite(state)):
                # Replace NaN/inf values with small random noise
                state = np.where(np.isfinite(state), state, np.random.normal(0, 0.01, state.shape))

            # Clamp state values to reasonable range
            state = np.clip(state, -10.0, 10.0)

            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            try:
                logits, value = self.model(state_t)

                # Ensure logits are valid for Categorical distribution
                if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                    logits = torch.zeros_like(logits)
                    logits[0, 0] = 1.0  # Default to first action

                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

                # Validate outputs
                if torch.isnan(logp) or torch.isinf(logp):
                    logp = torch.tensor(0.0, device=self.device)
                if torch.isnan(value) or torch.isinf(value):
                    value = torch.tensor(0.0, device=self.device)

            except Exception as e:
                logger.warning(f"Error in model forward pass: {e}")
                action = torch.tensor(0, device=self.device)  # Default action
                logp = torch.tensor(0.0, device=self.device)
                value = torch.tensor(0.0, device=self.device)

            obs_buf.append(state)
            act_buf.append(action.item())
            logp_buf.append(logp.item())
            val_buf.append(value.item())

            # *** USE STEP-LEVEL reward (not cumulative total_profit) ***
            next_state, reward, done, info = _unpack_step(self.env.step(action.item()))
            rew_buf.append(float(reward))
            done_buf.append(done)

            if not done:
                state = next_state
            else:
                state = _unpack_reset(self.env.reset())
                # Ensure reset state is not empty
                if len(state) == 0:
                    state = np.zeros(self.env.observation_space.shape, dtype=np.float32)

            # Update progress bar with current metrics
            if step % 100 == 0:
                trajectory_pbar.set_postfix({
                    'avg_reward': f"{np.mean(rew_buf[-100:]):.4f}" if rew_buf else "0.0000",
                    'episode_done': sum(done_buf[-100:]) if done_buf else 0
                })

        trajectory_pbar.close()
        return (
            np.array(obs_buf, dtype=np.float32),
            np.array(act_buf, dtype=np.int64),
            np.array(rew_buf, dtype=np.float32),
            np.array(logp_buf, dtype=np.float32),
            np.array(val_buf, dtype=np.float32),
            np.array(done_buf, dtype=np.bool_),
            state  # final state after rollout
        )

    def compute_gae(self, rewards, values, dones, last_state):
        """
        Compute advantages and discounted returns using GAE — all on the GPU.
        Args:
          rewards: np.ndarray of shape [T]
          values:  np.ndarray of shape [T]
          dones:   np.ndarray of shape [T], dtype=bool
          last_state: final state for bootstrapping
        Returns:
          adv_t: torch.Tensor [T] on self.device
          ret_t: torch.Tensor [T] on self.device
        """
        # Ensure inputs are finite and scale rewards to reasonable range
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=-1.0)
        rewards = np.clip(rewards, -10.0, 10.0)  # Clip rewards to prevent explosion
        
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
        values = np.clip(values, -100.0, 100.0)  # Allow larger range for values
        
        last_state = np.nan_to_num(last_state, nan=0.0, posinf=1.0, neginf=-1.0)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            last_state_t = torch.tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, last_val_t = self.model(last_state_t)
            last_val_t = last_val_t.squeeze()

            # Clamp last value to prevent extreme values
            last_val_t = torch.clamp(last_val_t, -100.0, 100.0)

        values_t = torch.cat([values_t, last_val_t[None]], dim=0)

        T = rewards_t.size(0)
        advantages = torch.zeros_like(rewards_t, device=self.device)
        gae = torch.tensor(0.0, device=self.device)

        for t in reversed(range(T)):
            mask = 1.0 - dones_t[t]
            delta = rewards_t[t] + self.gamma * values_t[t + 1] * mask - values_t[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae

            # Clamp GAE to prevent explosion but allow larger range
            gae = torch.clamp(gae, -50.0, 50.0)
            advantages[t] = gae

        returns = advantages + values_t[:-1]
        
        # Clamp returns to reasonable range
        returns = torch.clamp(returns, -100.0, 100.0)

        # More robust advantage normalization
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)

        if torch.isfinite(adv_mean) and torch.isfinite(adv_std) and adv_std > 1e-8:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            # Fallback normalization
            advantages = torch.clamp(advantages, -5.0, 5.0)

        return advantages, returns

    def train_step(self):
        """
        Perform one PPO update with comprehensive metrics and creative visualizations
        """
        try:
            # 1) rollout & GAE
            obs, acts, rews, old_lps, vals, dones, last_state = self.collect_trajectories()
            advs, rets = self.compute_gae(rews, vals, dones, last_state)

            # Check for NaN/infinite values in collected data
            if not np.all(np.isfinite(obs)):
                logger.warning("NaN/infinite observations detected, skipping update")
                return 0.0

            if not np.all(np.isfinite(rews)):
                logger.warning("NaN/infinite rewards detected, skipping update")
                return 0.0

            obs_t   = torch.as_tensor(obs,    dtype=torch.float32, device=self.device)
            acts_t  = torch.as_tensor(acts,   dtype=torch.long,   device=self.device)
            oldlp_t = torch.as_tensor(old_lps,dtype=torch.float32, device=self.device)
            adv_t   = advs  # Already on correct device from compute_gae
            ret_t   = rets  # Already on correct device from compute_gae

            # Track action distribution for analysis (ensure acts is numpy array)
            if isinstance(acts, torch.Tensor):
                acts_numpy = acts.detach().cpu().numpy()
            else:
                acts_numpy = np.array(acts)
            action_counts = np.bincount(acts_numpy, minlength=self.env.action_space.n)
            action_probs = action_counts / len(acts_numpy)
            self.action_distribution_history.append(action_probs)

            # Track reward patterns (ensure rews is numpy array)
            if isinstance(rews, torch.Tensor):
                rews_numpy = rews.detach().cpu().numpy()
            else:
                rews_numpy = np.array(rews)
            self.reward_trends.extend(rews_numpy)

            # Analyze trading behavior (ensure obs is numpy array)
            if isinstance(obs, torch.Tensor):
                obs_numpy = obs.detach().cpu().numpy()
            else:
                obs_numpy = np.array(obs)
            self._analyze_trading_behavior(acts_numpy, rews_numpy, obs_numpy)

            # 2) PPO epochs with enhanced tracking
            policy_losses, value_losses, entropies = [], [], []
            kl_divergences, clip_fractions = [], []

            dataset_size = len(obs)

            # Progress bar for PPO update epochs
            epoch_pbar = tqdm(range(self.update_epochs), 
                             desc="PPO Update Epochs", 
                             leave=False, 
                             disable=(self.local_rank != 0))

            for epoch in epoch_pbar:
                perm = torch.randperm(dataset_size, device=self.device)

                # Progress bar for batch processing within each epoch
                batch_starts = list(range(0, dataset_size, self.batch_size))
                batch_pbar = tqdm(batch_starts, 
                                 desc=f"Epoch {epoch+1} Batches", 
                                 leave=False, 
                                 disable=(self.local_rank != 0))

                for start in batch_pbar:
                    idx = perm[start:start + self.batch_size]
                    b_obs, b_acts = obs_t[idx], acts_t[idx]
                    b_oldlp, b_adv = oldlp_t[idx], adv_t[idx]
                    b_ret = ret_t[idx]

                    logits, value_pred = self.model(b_obs)
                    dist = Categorical(logits=logits)
                    new_lps = dist.log_prob(b_acts)
                    entropy = dist.entropy().mean()

                    # Calculate KL divergence and clipping metrics
                    ratio = torch.exp(new_lps - b_oldlp)
                    kl_div = torch.mean(b_oldlp - new_lps)
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float())

                    kl_divergences.append(kl_div.item())
                    clip_fractions.append(clip_frac.item())

                    # policy loss
                    clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    policy_loss = -torch.min(ratio * b_adv, clipped * b_adv).mean()

                    # value loss (clipped and weighted) - with better scaling
                    value_pred = value_pred.squeeze()
                    with torch.no_grad():
                        old_vals = torch.tensor(vals, dtype=torch.float32, device=self.device)[idx]
                    
                    # Clamp predictions to reasonable range
                    value_pred = torch.clamp(value_pred, -100.0, 100.0)
                    old_vals = torch.clamp(old_vals, -100.0, 100.0)
                    
                    v_clipped = old_vals + torch.clamp(
                        value_pred - old_vals,
                        -self.clip_epsilon, self.clip_epsilon
                    )
                    
                    # Use Huber loss instead of MSE for more stable training
                    def huber_loss(pred, target, delta=1.0):
                        residual = torch.abs(pred - target)
                        condition = residual <= delta
                        small_res = 0.5 * residual ** 2
                        large_res = delta * residual - 0.5 * delta ** 2
                        return torch.where(condition, small_res, large_res)
                    
                    value_loss1 = huber_loss(value_pred, b_ret, delta=10.0)
                    value_loss2 = huber_loss(v_clipped, b_ret, delta=10.0)
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                    
                    # Scale down value loss to be comparable to policy loss
                    value_loss = value_loss * 0.5

                    loss = policy_loss + value_loss - self.entropy_coef * entropy

                    # Check for NaN/infinite values in loss before backward pass
                    if not torch.isfinite(loss):
                        logger.warning(f"NaN/infinite loss detected: {loss.item()}, skipping update")
                        continue

                    # Check for NaN/infinite values in logits
                    if not torch.all(torch.isfinite(logits)):
                        logger.warning("NaN/infinite logits detected, skipping update")
                        continue

                    self.optimizer.zero_grad()
                    loss.backward()

                    # Calculate gradient norm with better clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                    # Check for NaN gradients
                    has_nan_grad = False
                    for param in self.model.parameters():
                        if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                            has_nan_grad = True
                            break

                    if has_nan_grad:
                        logger.warning("NaN gradients detected, skipping optimizer step")
                        self.optimizer.zero_grad()
                        continue

                    if not torch.isfinite(grad_norm):
                        logger.warning("NaN gradient norm detected, skipping optimizer step")
                        self.optimizer.zero_grad()
                        continue

                    self.optimizer.step()

                    # Verify model parameters are still valid after update
                    has_nan_params = False
                    for name, param in self.model.named_parameters():
                        if not torch.all(torch.isfinite(param)):
                            logger.error(f"NaN/infinite parameters detected in {name}, reinitializing layer")
                            has_nan_params = True
                            # Reinitialize the problematic layer with more conservative values
                            if hasattr(param, 'data'):
                                param.data.normal_(0, 0.001)  # Much smaller initialization

                    # If we had NaN parameters, also reset optimizer state
                    if has_nan_params:
                        logger.info("Resetting optimizer state due to NaN parameters")
                        self.optimizer.state = {}

                    # Store valid gradient norm
                    if torch.isfinite(grad_norm):
                        self.gradient_norms.append(grad_norm.item())

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.item())

                    # Update batch progress bar
                    batch_pbar.set_postfix({
                        'policy_loss': f"{policy_loss.item():.4f}",
                        'value_loss': f"{value_loss.item():.4f}",
                        'entropy': f"{entropy.item():.4f}"
                    })

                batch_pbar.close()

                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    'avg_policy_loss': f"{np.mean(policy_losses[-len(batch_starts):]):.4f}",
                    'avg_value_loss': f"{np.mean(value_losses[-len(batch_starts):]):.4f}",
                    'kl_div': f"{np.mean(kl_divergences[-len(batch_starts):]):.4f}"
                })

            epoch_pbar.close()

            # Store loss trends
            self.loss_trends.extend(policy_losses)

            # Calculate value function accuracy
            with torch.no_grad():
                try:
                    logits, value_preds = self.model(obs_t)
                    value_preds_squeezed = value_preds.squeeze()

                    # Ensure both tensors are on the same device
                    if value_preds_squeezed.device != ret_t.device:
                        value_preds_squeezed = value_preds_squeezed.to(ret_t.device)

                    # Calculate accuracy with NaN protection
                    abs_diff = torch.abs(value_preds_squeezed - ret_t)
                    abs_ret = torch.abs(ret_t) + 1e-8
                    relative_error = abs_diff / abs_ret

                    # Filter out NaN/infinite values
                    valid_mask = torch.isfinite(relative_error)
                    if valid_mask.any():
                        value_accuracy = 1.0 - torch.mean(relative_error[valid_mask])
                        if torch.isfinite(value_accuracy):
                            self.value_accuracy.append(float(value_accuracy.cpu().item()))
                        else:
                            self.value_accuracy.append(0.5)  # Default value
                    else:
                        self.value_accuracy.append(0.5)  # Default value
                except Exception as e:
                    logger.warning(f"Error calculating value accuracy: {e}")
                    self.value_accuracy.append(0.5)  # Default value

            # Enhanced logging
            self._log_comprehensive_metrics(
                policy_losses, value_losses, entropies, kl_divergences, 
                clip_fractions, action_probs, rews, vals, rets.detach().cpu().numpy()
            )

            # Create advanced visualizations
            if self.global_step % 50 == 0:
                self._create_advanced_visualizations()

            # decay LR & entropy once per train_step
            if getattr(self, "current_update", 0) % 50 == 0:
                self.scheduler.step()
            self.entropy_coef = max(0.001, self.entropy_coef * 0.9995)  # Slower decay with minimum

            # rollout reward: total sum of step rewards
            total_reward = float(rews.sum())

        except Exception as e:
            logger.error(f"Error in train_step: {e}")
            return 0.0

        # checkpoint with backup (save much less frequently to reduce log spam)
        if self.local_rank == 0 and getattr(self, "current_update", 0) % 200 == 0 and getattr(self, "current_update", 0) > 0:
            state_dict = (
                self.model.module.state_dict()
                if isinstance(self.model, DDP)
                else self.model.state_dict()
            )
            ckpt = {
                "update_idx": getattr(self, "current_update", 0),
                "model_state": state_dict,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "entropy_coef": self.entropy_coef,
                "total_reward": total_reward,
                "timestamp": datetime.datetime.now().isoformat()
            }

            # Save checkpoint
            ckpt_path = self.model_save_path + ".ckpt"
            torch.save(ckpt, ckpt_path)

            # Save model state separately for inference
            model_for_save = self.model.module if isinstance(self.model, DDP) else self.model
            if hasattr(model_for_save, 'save_model'):
                model_for_save.save_model(self.model_save_path)
            else:
                torch.save(state_dict, self.model_save_path)

            # Create backup every 500 updates
            if hasattr(self, "current_update") and self.current_update % 500 == 0:
                backup_name = f"{self.model_save_path}.backup_{self.current_update}"
                torch.save(ckpt, backup_name)

        self.global_step += 1
        return total_reward

    def train(
        self,
        total_timesteps: int,
        start_update: int = 0,
        eval_env=None,
    ):
        """
        Run PPO until `total_timesteps` env steps are collected.
        Only rank 0 displays a single-line tqdm bar updated in place.
        """
        n_updates = max(1, total_timesteps // self.rollout_steps)
        logger.info(f"Starting training: updates {start_update} → {n_updates - 1}")

        # log hyperparameters once
        self.tb_writer.add_hparams(
            {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "rollout_steps": self.rollout_steps,
                "update_epochs": self.update_epochs,
                "batch_size": self.batch_size,
                "lr": self.optimizer.param_groups[0]["lr"]
            },
            {"rollout_reward": 0}
        )

        start_time = time.time()
        if eval_env is None:
            logger.warning("No evaluation environment provided; skipping eval logs")

        if self.local_rank == 0:
            pbar = trange(start_update, n_updates, desc="PPO updates")
        else:
            pbar = range(start_update, n_updates)

        for update in pbar:
            self.current_update = update
            mean_reward = self.train_step()

            if self.local_rank == 0:
                pbar.set_postfix(total_reward=f"{mean_reward:.4f}")

            elapsed = time.time() - start_time
            self.tb_writer.add_scalar("PPO/ElapsedSeconds", elapsed, update)

            # periodic evaluation
            if (
                eval_env is not None
                and self.local_rank == 0
                and (update + 1) % self.eval_interval == 0
            ):
                real_agent = self.model.module if isinstance(self.model, DDP) else self.model
                profits, times = evaluate_agent_distributed(eval_env, real_agent, 0)
                cagr, sharpe, mdd = compute_performance_metrics(profits, times)
                self.tb_writer.add_scalar("PPO/Eval/CAGR", cagr, update)
                self.tb_writer.add_scalar("PPO/Eval/Sharpe", sharpe, update)
                self.tb_writer.add_scalar("PPO/Eval/MDD", mdd, update)

                fig = plt.figure()
                pd.Series(profits).cumsum().plot(
                    title=f"PPO Equity after update {update + 1}"
                )
                self.tb_writer.add_figure("PPO/Eval/EquityCurve", fig, update)
                plt.close(fig)

            # periodic weight histograms
            if self.local_rank == 0 and (update + 1) % (self.eval_interval * 2) == 0:
                real_model = self.model.module if isinstance(self.model, DDP) else self.model

    def _log_comprehensive_metrics(self, policy_losses, value_losses, entropies, kl_divergences, clip_fractions, action_probs, rews, vals, rets):
        """Log detailed metrics to TensorBoard."""
        self.tb_writer.add_scalar("Losses/PolicyLoss", np.mean(policy_losses), self.global_step)
        self.tb_writer.add_scalar("Losses/ValueLoss", np.mean(value_losses), self.global_step)
        self.tb_writer.add_scalar("Losses/Entropy", np.mean(entropies), self.global_step)
        self.tb_writer.add_scalar("PPO/KL_Divergence", np.mean(kl_divergences), self.global_step)
        self.tb_writer.add_scalar("PPO/Clip_Fraction", np.mean(clip_fractions), self.global_step)
        self.tb_writer.add_scalar("PPO/LearningRate", self.optimizer.param_groups[0]["lr"], self.global_step)
        self.tb_writer.add_scalar("PPO/EntropyCoefficient", self.entropy_coef, self.global_step)

        # Distribution of actions
        action_labels = [f"action_{i}" for i in range(len(action_probs))]
        self.tb_writer.add_scalars("Actions/Distribution", dict(zip(action_labels, action_probs)), self.global_step)

        # More detailed stats
        self.tb_writer.add_scalar("Rewards/Mean", np.mean(rews), self.global_step)
        self.tb_writer.add_scalar("Rewards/Std", np.std(rews), self.global_step)
        self.tb_writer.add_scalar("Values/Mean", np.mean(vals), self.global_step)
        self.tb_writer.add_scalar("Values/Std", np.std(vals), self.global_step)
        self.tb_writer.add_scalar("Returns/Mean", np.mean(rets), self.global_step)
        self.tb_writer.add_scalar("Returns/Std", np.std(rets), self.global_step)

        # Value function accuracy
        if self.value_accuracy:
            self.tb_writer.add_scalar("Values/Accuracy", np.mean(self.value_accuracy), self.global_step)

        # Gradient Norm
        if self.gradient_norms:
            self.tb_writer.add_scalar("Gradients/Norm", np.mean(self.gradient_norms), self.global_step)

        # Policy Divergence
        if self.policy_divergence:
            self.tb_writer.add_scalar("PPO/PolicyDivergence", np.mean(self.policy_divergence), self.global_step)

    def _create_advanced_visualizations(self):
        """Create more complex plots and visualizations for deeper insights."""
        # Reward Trend over Time
        if self.reward_trends:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=range(len(self.reward_trends)), y=self.reward_trends, ax=ax)
            ax.set_title("Reward Trend Over Time")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            self.tb_writer.add_figure("Visualizations/RewardTrend", fig, self.global_step)
            plt.close(fig)

        # Loss Trend over Time
        if self.loss_trends:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=range(len(self.loss_trends)), y=self.loss_trends, ax=ax)
            ax.set_title("Loss Trend Over Time")
            ax.set_xlabel("Update Step")
            ax.set_ylabel("Loss")
            self.tb_writer.add_figure("Visualizations/LossTrend", fig, self.global_step)
            plt.close(fig)

        # Action Distribution Heatmap
        if self.action_distribution_history:
            action_matrix = np.vstack(self.action_distribution_history)
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(action_matrix, cmap="viridis", ax=ax, cbar_kws={'label': 'Probability'})
            ax.set_title("Action Distribution Over Time")
            ax.set_xlabel("Action")
            ax.set_ylabel("Time Step")
            self.tb_writer.add_figure("Visualizations/ActionDistribution", fig, self.global_step)
            plt.close(fig)

    def _analyze_trading_behavior(self, actions, rewards, observations):
        """
        Analyze trading-specific metrics such as position changes, profit per trade, and drawdown periods.
        Args:
            actions: array of actions taken
            rewards: array of rewards received
            observations: array of states observed
        """
        # This is a placeholder; trading behavior analysis is environment-specific
        pass

    def get_training_metrics(self, policy_losses, value_losses, entropies, kl_divergences):
        """Get training metrics dictionary"""
        metrics = {
            'avg_policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
            'avg_value_loss': np.mean(value_losses) if value_losses else 0.0,
            'avg_entropy': np.mean([entropy.item() for entropy in entropies]) if entropies else 0.0,
            'kl_div': np.mean(kl_divergences) if kl_divergences else 0.0
        }
        return metrics