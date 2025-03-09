import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import os
import cudf
import time
import datetime
from data_preprocessing import create_environment_data
from trading_environment import TradingEnvironment
from ga_policy_evolution import run_ga_evolution, PolicyNetwork
from policy_gradient_methods import PPOTrainer


def compute_performance_metrics(balance_history):
    """
    Compute performance metrics: CAGR, Sharpe Ratio, and Maximum Drawdown.

    Args:
        balance_history (list): List of balance values over time.

    Returns:
        tuple: (CAGR, Sharpe Ratio, Maximum Drawdown)
    """
    steps_per_year = 252 * 390  # Trading minutes in a year
    total_return = (balance_history[-1] - balance_history[0]) / balance_history[0]
    years = len(balance_history) / steps_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    returns = np.diff(balance_history) / balance_history[:-1]
    ann_mean = np.mean(returns) * steps_per_year
    ann_std = np.std(returns) * np.sqrt(steps_per_year)
    sharpe = ann_mean / (ann_std + 1e-8)  # Add small epsilon to avoid division by zero
    running_max = np.maximum.accumulate(balance_history)
    mdd = np.max((running_max - balance_history) / running_max)
    return cagr, sharpe, mdd


def evaluate_agent(env, agent, steps):
    """
    Evaluate an agent's performance in the trading environment.

    Args:
        env (TradingEnvironment): The trading environment instance.
        agent: The policy network or PPO trainer model.
        steps (int): Number of steps to evaluate.

    Returns:
        list: Balance history over the evaluation period.
    """
    balance_history = []
    obs = env.reset()
    device = next(agent.parameters()).device
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    for _ in range(steps):
        if env.done:
            break

        with torch.no_grad():
            if hasattr(agent, 'act'):
                action = agent.act(obs_tensor)
            else:
                policy_logits, _ = agent.forward(obs_tensor)
                dist_ = torch.distributions.Categorical(logits=policy_logits)
                action = dist_.sample().item()

        obs, _, done, _ = env.step(action)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        balance_history.append(env.current_balance)

        if done:
            break

    return balance_history


def main():
    """
    Main function to orchestrate distributed training with GA and PPO policies.
    Sets a high timeout to prevent NCCL watchdog issues and includes profiling.
    """
    timeout = datetime.timedelta(seconds=3600)
    # Initialize distributed process group with a 1-hour timeout to prevent watchdog issues
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timeout)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Define data paths
    data_folder = './data_txt'
    cache_folder = './cached_data'
    os.makedirs(cache_folder, exist_ok=True)
    train_data_path = f'{cache_folder}/train_data.parquet'
    test_data_path = f'{cache_folder}/test_data.parquet'

    # Data preparation (rank 0 only)
    if local_rank == 0:
        if not (os.path.exists(train_data_path) and os.path.exists(test_data_path)):
            train_data, test_data, scaler = create_environment_data(
                data_folder=data_folder,
                max_rows=None,
                use_gpu=True,
                cache_folder=cache_folder
            )
            train_data.to_parquet(train_data_path)
            test_data.to_parquet(test_data_path)
            print("✅ Data cached successfully.")
        else:
            print("✅ Cached data already exists.")

    # Wait for rank 0 to finish data preparation
    dist.barrier()

    # Load cached data on all ranks
    train_data = cudf.read_parquet(train_data_path).to_pandas()
    test_data = cudf.read_parquet(test_data_path).to_pandas()

    # Initialize environments
    train_env = TradingEnvironment(train_data, initial_balance=100000.0)
    test_env = TradingEnvironment(test_data, initial_balance=100000.0)

    # GA training setup
    ga_model_path = "ga_policy_model.pth"
    ga_agent = PolicyNetwork(train_env.observation_dim, 64, train_env.action_space.n, device=device)

    # Distributed GA training
    start_time = time.time()
    if not os.path.exists(ga_model_path):
        if local_rank == 0:
            # Only rank 0 updates ga_agent and saves the model
            ga_agent, best_fit = run_ga_evolution(
                train_env,
                population_size=40,
                generations=12,
                device=device,
                model_save_path=ga_model_path,
                distributed=True,
                local_rank=local_rank,
                world_size=world_size
            )
            print(f"GA best fitness: {best_fit:.2f}")
        else:
            # Non-zero ranks participate in distributed computation but don’t update ga_agent
            run_ga_evolution(
                train_env,
                population_size=40,
                generations=12,
                device=device,
                model_save_path=ga_model_path,
                distributed=True,
                local_rank=local_rank,
                world_size=world_size
            )

    # Synchronize all ranks after GA evolution
    dist.barrier()

    # Reinitialize ga_agent on all ranks and load the saved model
    ga_agent = PolicyNetwork(train_env.observation_dim, 64, train_env.action_space.n, device=device)
    ga_agent.load_model(ga_model_path)

    ga_training_time = time.time() - start_time
    if local_rank == 0:
        print(f"GA training took {ga_training_time:.2f} seconds")

    # Evaluate GA agent (rank 0 only)
    if local_rank == 0:
        balance_history_ga = evaluate_agent(test_env, ga_agent, len(test_data))
        cagr_ga, sharpe_ga, mdd_ga = compute_performance_metrics(balance_history_ga)
        print(f"GA Results - CAGR: {cagr_ga:.4f}, Sharpe: {sharpe_ga:.4f}, MaxDD: {mdd_ga:.4f}")

    # PPO training setup
    ppo_model_path = "ppo_actor_critic_model.pth"
    # Initialize PPOTrainer
    ppo_trainer = PPOTrainer(
        env=train_env,
        input_dim=train_env.observation_space.shape[0],  # Observation dimension
        action_dim=train_env.action_space.n,  # Number of actions
        hidden_dim=64,  # Network size
        lr=3e-4,  # Learning rate
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE smoothing
        clip_epsilon=0.2,  # PPO clipping
        update_epochs=4,  # Epochs per update
        rollout_steps=2048,  # Steps per rollout
        batch_size=64,  # Mini-batch size
        device=device,  # GPU device
        model_save_path=ppo_model_path,  # Save path
        local_rank=local_rank  # Distributed rank
    )

    if local_rank == 0:
        start_time = time.time()
        if not os.path.exists(ppo_model_path):
            ppo_trainer.train(total_timesteps=1000000)
            ppo_trainer.model.save_model(ppo_model_path)
        else:
            ppo_trainer.model.load_model(ppo_model_path)
            print("Loaded existing PPO model.")
        ppo_training_time = time.time() - start_time
        print(f"PPO training took {ppo_training_time:.2f} seconds")

    # Synchronize after PPO training
    dist.barrier()
    ppo_trainer.model.load_model(ppo_model_path)

    # Evaluate PPO agent and plot results (rank 0 only)
    if local_rank == 0:
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

    # Clean up distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()