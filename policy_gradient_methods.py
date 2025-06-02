# policy_gradient_methods.py

"""
Proximal Policy Optimization (PPO) implementation using PyTorch and DistributedDataParallel.

This module defines:
  - ActorCriticNet: shared-base actor-critic network
  - PPOTrainer: collects trajectories, computes GAE, and updates policy/value heads
  - Extensive TensorBoard instrumentation for losses, rewards, LR, entropy, and periodic evaluation metrics
"""

import os
import datetime
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from utils import evaluate_agent_distributed, compute_performance_metrics

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)


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
        # Shared base network
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

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Args:
            x: tensor of shape [batch, input_dim] or higher dimensions
        Returns:
            policy_logits: [batch, action_dim]
            value: [batch, 1]
        """
        x = x.to(self.device)
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        hidden = self.base(x)
        return self.policy_head(hidden), self.value_head(hidden)

    def act(self, state):
        """
        Select action greedily (for evaluation).
        Args:
            state: numpy array or torch tensor, shape [input_dim]
        Returns:
            action: int
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.to(self.device)
        with torch.no_grad():
            logits, _ = self.forward(state)
            return torch.argmax(logits, dim=-1).item()

    def save_model(self, path: str):
        """Save model state dict."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Saved model to {path} at {datetime.datetime.now()}")

    def load_model(self, path: str):
        """Load model state dict if exists, else leave random init."""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Loaded model from {path}")
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

        # Optimizer & LR scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        # Entropy coefficient for exploration
        self.entropy_coef = 0.05

        # TensorBoard writer (one logdir per rank)
        run_dir = f"./runs/ppo_rank{self.local_rank}"
        self.tb_writer = SummaryWriter(log_dir=run_dir, flush_secs=5)
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

        state = self.env.reset()
        for _ in range(self.rollout_steps):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits, value = self.model(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

            obs_buf.append(state)
            act_buf.append(action.item())
            logp_buf.append(logp.item())
            val_buf.append(value.item())

            next_state, reward, done, _ = self.env.step(action.item())
            rew_buf.append(reward)
            done_buf.append(done)

            state = next_state if not done else self.env.reset()

        return (
            np.array(obs_buf, dtype=np.float32),
            np.array(act_buf, dtype=np.int64),
            np.array(rew_buf, dtype=np.float32),
            np.array(logp_buf, dtype=np.float32),
            np.array(val_buf, dtype=np.float32),
            np.array(done_buf, dtype=np.bool_)
        )

    def compute_gae(self, rewards, values, dones):
        """
        Compute advantages and discounted returns using GAE.
        """
        # bootstrap last value
        state = self.env.reset()
        last_val = self.model(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        )[1].item()
        values = np.append(values, last_val)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def train_step(self):
        """
        Perform one PPO update:
          - collect rollouts
          - compute GAE & returns
          - multiple minibatch updates
          - log losses, entropy, and LR
          - save checkpoint
        Returns mean rollout reward.
        """
        # 1) rollout
        obs, acts, rews, old_lps, vals, dones = self.collect_trajectories()
        advs, rets = self.compute_gae(rews, vals, dones)

        # tensors
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        acts_t = torch.tensor(acts, dtype=torch.long).to(self.device)
        oldlp_t = torch.tensor(old_lps, dtype=torch.float32).to(self.device)
        adv_t = torch.tensor(advs, dtype=torch.float32).to(self.device)
        ret_t = torch.tensor(rets, dtype=torch.float32).to(self.device)

        dataset_size = len(obs)
        # 2) PPO epochs
        for epoch in range(self.update_epochs):
            perm = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, self.batch_size):
                idx = perm[start:start + self.batch_size]
                b_obs, b_acts = obs_t[idx], acts_t[idx]
                b_oldlp, b_adv = oldlp_t[idx], adv_t[idx]
                b_ret = ret_t[idx]

                logits, value_pred = self.model(b_obs)
                dist = Categorical(logits=logits)
                new_lps = dist.log_prob(b_acts)
                entropy = dist.entropy().mean()

                # policy loss
                ratio = torch.exp(new_lps - b_oldlp)
                clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(ratio * b_adv, clipped * b_adv).mean()

                # value loss
                value_pred = value_pred.squeeze()
                with torch.no_grad():
                    old_vals = torch.tensor(vals, dtype=torch.float32, device=self.device)[idx]
                v_clipped = old_vals + torch.clamp(value_pred - old_vals,
                                                   -self.clip_epsilon, self.clip_epsilon)
                value_loss = 0.5 * torch.max((value_pred - b_ret) ** 2,
                                             (v_clipped - b_ret) ** 2).mean()

                # total loss
                loss = policy_loss + value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                # log scalars
                self.tb_writer.add_scalar("ppo/policy_loss", policy_loss.item(), self.global_step)
                self.tb_writer.add_scalar("ppo/value_loss", value_loss.item(), self.global_step)
                self.tb_writer.add_scalar("ppo/entropy", entropy.item(), self.global_step)
                self.tb_writer.add_scalar("ppo/lr", self.scheduler.get_last_lr()[0], self.global_step)
                self.global_step += 1

            # decay LR & entropy
            self.scheduler.step()
            self.entropy_coef *= 0.95

        # 3) rollout reward
        mean_reward = float(rews.mean())
        self.tb_writer.add_scalar("ppo/rollout_reward", mean_reward, self.global_step)

        # 4) checkpoint
        if self.local_rank == 0:
            state_dict = (self.model.module.state_dict()
                          if isinstance(self.model, DDP)
                          else self.model.state_dict())
            ckpt = {
                "update_idx": getattr(self, "current_update", 0),
                "model_state": state_dict,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "entropy_coef": self.entropy_coef
            }
            torch.save(ckpt, self.model_save_path + ".ckpt")

        logger.info(f"PPO update complete — mean reward: {mean_reward:.4f}")
        return mean_reward

    def train(self, total_timesteps: int, start_update: int = 0):
        """
        Run PPO until `total_timesteps` env steps are collected.
        Also performs periodic evaluation and logs profitability metrics,
        with a tqdm progress bar and elapsed‐time scalars.
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

        # for elapsed time
        start_time = time.time()

        # NOTE: assumes `test_env` and `evaluate_agent_distributed` are in scope
        from main import test_env

        for update in trange(start_update, n_updates, desc="PPO updates"):
            self.current_update = update

            # 1) do one PPO update
            mean_reward = self.train_step()
            logger.info(f"Update {update + 1}/{n_updates} done — mean reward: {mean_reward:.4f}")

            # 2) log elapsed time
            elapsed = time.time() - start_time
            self.tb_writer.add_scalar("PPO/ElapsedSeconds", elapsed, update)

            # 3) periodic evaluation & profit metrics
            if self.local_rank == 0 and (update + 1) % self.eval_interval == 0:
                profits, times = evaluate_agent_distributed(test_env, self.model, 0)
                cagr, sharpe, mdd = compute_performance_metrics(profits, times)
                self.tb_writer.add_scalar("PPO/Eval/CAGR", cagr, update)
                self.tb_writer.add_scalar("PPO/Eval/Sharpe", sharpe, update)
                self.tb_writer.add_scalar("PPO/Eval/MDD", mdd, update)

                # log equity curve image
                fig = plt.figure()
                pd.Series(profits).cumsum().plot(
                    title=f"PPO Equity after update {update + 1}"
                )
                self.tb_writer.add_figure("PPO/Eval/EquityCurve", fig, update)
                plt.close(fig)

            # 4) periodic weight histograms
            if self.local_rank == 0 and (update + 1) % (self.eval_interval * 2) == 0:
                for i, layer in enumerate(self.model.base):
                    if isinstance(layer, nn.Linear):
                        w = layer.weight.data.cpu().numpy()
                        self.tb_writer.add_histogram(f"PPO/Layer{i}_Weights", w, update)

        # finalize
        self.tb_writer.close()
        return self.model.module if isinstance(self.model, DDP) else self.model
