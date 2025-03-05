import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from data_preprocessing import create_environment_data
from trading_environment import TradingEnvironment
from ga_policy_evolution import run_ga_evolution, PolicyNetwork
from policy_gradient_methods import PPOTrainer


def compute_performance_metrics(balance_history, risk_free_rate=0.0):
    """
    Compute performance metrics from balance history:
    - Compound Annual Growth Rate (CAGR)
    - Sharpe Ratio
    - Maximum Drawdown (MaxDD)
    """
    steps_per_year = 252 * 390  # assuming 1-min bars
    total_return = (balance_history[-1] - balance_history[0]) / balance_history[0]
    years = len(balance_history) / steps_per_year
    cagr = (1 + total_return)**(1 / years) - 1 if years > 0 else 0
    returns = np.diff(balance_history) / balance_history[:-1]
    ann_mean = np.mean(returns) * steps_per_year
    ann_std = np.std(returns) * np.sqrt(steps_per_year)
    sharpe = ann_mean / (ann_std + 1e-8)
    running_max = np.maximum.accumulate(balance_history)
    mdd = np.max((running_max - balance_history) / running_max)
    return cagr, sharpe, mdd


def evaluate_agent(env, agent, steps):
    """
    Evaluate a trained agent (GA or PPO) and return the balance history.
    """
    balance_history = []
    obs = env.reset()
    device = next(agent.parameters()).device
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

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
    # Data preprocessing
    data_folder = './data_txt'
    train_data, test_data, scaler = create_environment_data(data_folder, max_rows=100000)

    # Create training and test environments
    train_env = TradingEnvironment(train_data, initial_balance=100000.0)
    test_env = TradingEnvironment(test_data, initial_balance=100000.0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Genetic Algorithm (GA) model loading or training
    ga_model_path = "ga_policy_model.pth"
    ga_agent = PolicyNetwork(train_env.observation_dim, 64, train_env.action_space, device=device)

    if os.path.exists(ga_model_path):
        ga_agent.load_model(ga_model_path)
        print("GA model loaded successfully.")
    else:
        ga_agent, best_fit = run_ga_evolution(train_env, population_size=40, generations=12, device=device)
        ga_agent.save_model(ga_model_path)
        print(f"GA best fitness achieved: {best_fit:.2f}")

    # Evaluate GA agent
    balance_history_ga = evaluate_agent(test_env, ga_agent, len(test_data))
    cagr_ga, sharpe_ga, mdd_ga = compute_performance_metrics(balance_history_ga)
    print(f"GA Results - CAGR: {cagr_ga:.4f}, Sharpe: {sharpe_ga:.4f}, MaxDD: {mdd_ga:.4f}")

    # PPO model loading or training
    ppo_model_path = "ppo_actor_critic_model.pth"
    ppo_trainer = PPOTrainer(train_env,
                             input_dim=train_env.observation_dim,
                             action_dim=train_env.action_space,
                             hidden_dim=64,
                             lr=3e-4,
                             gamma=0.99,
                             clip_epsilon=0.2,
                             update_epochs=10,
                             rollout_steps=500,
                             device=device,
                             model_save_path=ppo_model_path)

    if os.path.exists(ppo_model_path):
        ppo_trainer.model.load_model(ppo_model_path)
        print("PPO model loaded successfully.")
    else:
        ppo_trainer.train(total_timesteps=100000)
        ppo_trainer.model.save_model(ppo_model_path)

    # Evaluate PPO agent
    balance_history_ppo = evaluate_agent(test_env, ppo_trainer.model, len(test_data))
    cagr_ppo, sharpe_ppo, mdd_ppo = compute_performance_metrics(balance_history_ppo)
    print(f"PPO Results - CAGR: {cagr_ppo:.4f}, Sharpe: {sharpe_ppo:.4f}, MaxDD: {mdd_ppo:.4f}")

    # Plot equity curves
    plt.figure(figsize=(10, 6))
    plt.plot(balance_history_ga, label='GA Policy')
    plt.plot(balance_history_ppo, label='PPO Policy')
    plt.title('Equity Curves Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Balance')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

