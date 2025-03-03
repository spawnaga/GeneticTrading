import torch
import numpy as np
import pandas as pd
from data_preprocessing import create_environment_data
from trading_environment import TradingEnvironment
from ga_policy_evolution import run_ga_evolution
from policy_gradient_methods import PPOTrainer
import matplotlib.pyplot as plt

def compute_performance_metrics(balance_history, risk_free_rate=0.0):
    """
    Compute CAGR, Sharpe ratio, and max drawdown from a balance history.
    balance_history: array-like, daily or per-step equity values
    We assume each step ~ 1 minute, so annualization might differ. We'll do a rough approach:
    - If 1 minute data, then ~ 390 minutes per trading day => ~ 252 * 390 = ~ 98280 minutes/year
    - For real accuracy, you'd convert steps to actual time from date_time.
    """
    if len(balance_history) < 2:
        return 0.0, 0.0, 0.0

    initial_balance = balance_history[0]
    final_balance = balance_history[-1]
    total_return = (final_balance - initial_balance) / initial_balance

    # Approx steps in a year if 1-min bars
    steps_per_year = 252 * 390
    n_steps = len(balance_history)
    years = n_steps / steps_per_year

    if years <= 0:
        years = 1e-6

    cagr = (1 + total_return) ** (1 / years) - 1

    # Compute daily returns for Sharpe. We'll do minute returns for rough approximation.
    returns = np.diff(balance_history) / balance_history[:-1]
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)

    # Annualized stats
    ann_mean = mean_ret * steps_per_year
    ann_std = std_ret * np.sqrt(steps_per_year)

    if ann_std < 1e-8:
        sharpe = 0.0
    else:
        sharpe = (ann_mean - risk_free_rate) / ann_std

    # Max Drawdown
    # 1. Compute running maximum
    running_max = np.maximum.accumulate(balance_history)
    drawdowns = (running_max - balance_history) / running_max
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0  # ✅ Ensure MDD is properly computed

    return cagr, sharpe, max_drawdown


def evaluate_agent(env, agent, steps=10000):
    """
    Evaluate a trained agent (DQN-like or GA policy or PPO actor-critic).
    Return the final balance history for performance metrics.
    """
    device = next(agent.parameters()).device  # ✅ Get model device
    balance_history = []
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # ✅ Move obs to model's device

    for _ in range(steps):
        if env.done:
            break

        if hasattr(agent, 'act'):
            # ✅ Ensure state tensor is on the correct device
            action = agent.act(obs)
        else:
            # ✅ Move observation to the same device as the agent
            with torch.no_grad():
                policy_logits, _ = agent.forward(obs)  # ✅ `obs` is now on `device`
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample().item()

        obs, _, done, _ = env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # ✅ Keep new state on the same device
        balance_history.append(env.current_balance)

        if done:
            break

    return balance_history


def main():
    # Load data (limit to 1000 rows for testing)
    data_folder = './data_txt'
    train_data, test_data, scaler = create_environment_data(data_folder, max_rows=None)

    # Create training and test environment
    train_env = TradingEnvironment(train_data, initial_balance=100000.0)
    test_env = TradingEnvironment(test_data, initial_balance=100000.0)

    # ✅ GA Evolution (Test Faster)
    ga_agent, best_fit = run_ga_evolution(train_env, population_size=10, generations=3)
    print(f"GA best train fitness = {best_fit:.2f}")

    # Evaluate GA on test set
    test_env.reset()
    balance_history_ga = evaluate_agent(test_env, ga_agent, steps=len(test_data))
    cagr_ga, sharpe_ga, mdd_ga = compute_performance_metrics(balance_history_ga)
    print(f"GA Results - CAGR: {cagr_ga:.4f}, Sharpe: {sharpe_ga:.4f}, MaxDD: {mdd_ga:.4f}")

    # ✅ PPO Training (Reduce Steps for Faster Testing)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ppo_trainer = PPOTrainer(train_env,
                             input_dim=train_env.observation_dim,
                             action_dim=train_env.action_space,
                             hidden_dim=64,
                             lr=3e-4,
                             gamma=0.99,
                             clip_epsilon=0.2,
                             update_epochs=2,   # 🔥 Reduce epochs for faster training
                             rollout_steps=500,  # 🔥 Reduce rollout steps
                             device=device)
    ppo_model = ppo_trainer.train(total_timesteps=1000)  # 🔥 Reduce timesteps

    # Evaluate PPO on test set
    test_env.reset()
    balance_history_ppo = evaluate_agent(test_env, ppo_model, steps=len(test_data))
    cagr_ppo, sharpe_ppo, mdd_ppo = compute_performance_metrics(balance_history_ppo)
    print(f"PPO Results - CAGR: {cagr_ppo:.4f}, Sharpe: {sharpe_ppo:.4f}, MaxDD: {mdd_ppo:.4f}")

    # Plot equity curves
    plt.figure(figsize=(10,6))
    plt.plot(balance_history_ga, label='GA Policy')
    plt.plot(balance_history_ppo, label='PPO Policy')
    plt.title('Equity Curves Comparison')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()