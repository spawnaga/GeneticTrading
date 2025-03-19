import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import cudf
import time
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP  # Added for PPO distributed training
from data_preprocessing import create_environment_data
from trading_environment import TradingEnvironment
from ga_policy_evolution import run_ga_evolution, PolicyNetwork
from policy_gradient_methods import PPOTrainer, ActorCriticNet


def compute_performance_metrics(balance_history):
    """
    Compute key performance metrics for a trading strategy.

    Args:
        balance_history (list): List of account balance values over time.

    Returns:
        tuple: (CAGR, Sharpe Ratio, Maximum Drawdown) as floats.
    """
    steps_per_year = 252 * 390  # Trading minutes in a year (252 days * 390 minutes/day)
    total_return = (balance_history[-1] - balance_history[0]) / balance_history[0]
    years = len(balance_history) / steps_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    returns = np.diff(balance_history) / balance_history[:-1]
    ann_mean = np.mean(returns) * steps_per_year
    ann_std = np.std(returns) * np.sqrt(steps_per_year)
    sharpe = ann_mean / (ann_std + 1e-8)  # Small epsilon to avoid division by zero
    running_max = np.maximum.accumulate(balance_history)
    mdd = np.max((running_max - balance_history) / running_max)
    return cagr, sharpe, mdd


def evaluate_agent_distributed(env, agent, local_rank, world_size):
    """
    Evaluate an agent's performance across distributed ranks.

    Args:
        env (TradingEnvironment): Rank-specific trading environment instance.
        agent (nn.Module): Policy network or PPO model to evaluate.
        local_rank (int): Rank of the current process in the distributed group.
        world_size (int): Total number of processes (GPUs) in the group.

    Returns:
        list: Balance history for the rank (full history on rank 0, local on others).
    """
    balance_history = []
    obs = env.reset()
    device = next(agent.parameters()).device
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    while not env.done:
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

    # Gather variable-length balance histories from all ranks to rank 0
    balance_tensor = torch.tensor(balance_history, dtype=torch.float32).to(device)
    gather_list = [torch.zeros(len(balance_history), dtype=torch.float32).to(device) for _ in range(world_size)] if local_rank == 0 else None
    dist.gather(balance_tensor, gather_list=gather_list if local_rank == 0 else None, dst=0)

    if local_rank == 0:
        full_balance_history = torch.cat(gather_list).cpu().numpy().tolist()
        return full_balance_history
    return balance_history


def main():
    """
    Orchestrate distributed training and evaluation of Genetic Algorithm (GA) and Proximal Policy Optimization (PPO) policies.

    This function sets up a distributed training pipeline using PyTorch's NCCL backend for GPU communication.
    Each rank (GPU) processes a shard of the training and testing data, trains models in parallel, and evaluates
    their performance. Key features include:
    - Data sharding across ranks to ensure balanced workloads.
    - Distributed GA evolution with fitness evaluation across GPUs.
    - PPO training with DDP for synchronized gradient updates.
    - Performance metrics computation and visualization on rank 0.

    A 1-hour timeout is set to prevent NCCL watchdog issues during long-running operations.
    """
    # Initialize distributed environment with NCCL backend
    timeout = datetime.timedelta(seconds=3600)
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timeout)
    local_rank = int(os.environ["LOCAL_RANK"])  # Rank of this process within the node
    world_size = dist.get_world_size()          # Total number of processes (GPUs)
    torch.cuda.set_device(local_rank)           # Assign this process to its GPU
    device = torch.device(f"cuda:{local_rank}") # Device for model and tensor operations

    # Define paths for data caching
    data_folder = './data_txt'
    cache_folder = './cached_data'
    os.makedirs(cache_folder, exist_ok=True)    # Create cache directory if it doesn’t exist
    train_data_path = os.path.join(cache_folder, 'train_data.parquet')
    test_data_path = os.path.join(cache_folder, 'test_data.parquet')

    # Data preparation and caching (performed by rank 0 only)
    if local_rank == 0:
        if not (os.path.exists(train_data_path) and os.path.exists(test_data_path)):
            train_data, test_data, scaler = create_environment_data(
                data_folder=data_folder,
                max_rows=None,          # Use only the last 1000 rows of each split
                use_gpu=True,           # Leverage GPU acceleration for data processing
                cache_folder=cache_folder
            )
            train_data.to_parquet(train_data_path)
            test_data.to_parquet(test_data_path)
            print("✅ Data cached successfully.")
        else:
            print("✅ Cached data already exists.")
    dist.barrier()  # Synchronize all ranks after data preparation

    # Load cached data into memory on all ranks
    train_data = cudf.read_parquet(train_data_path).to_pandas()
    test_data = cudf.read_parquet(test_data_path).to_pandas()
    if local_rank == 0:
        print(f"Total train data size: {len(train_data)}, Total test data size: {len(test_data)}")

    # Shard data across ranks to balance workload
    train_chunk_size = max(1, len(train_data) // world_size)  # Ensure at least 1 row per rank
    test_chunk_size = max(1, len(test_data) // world_size)
    train_start = local_rank * train_chunk_size
    train_end = min((local_rank + 1) * train_chunk_size, len(train_data))
    test_start = local_rank * test_chunk_size
    test_end = min((local_rank + 1) * test_chunk_size, len(test_data))

    # Extract and validate data chunks for this rank
    train_chunk = train_data[train_start:train_end]
    test_chunk = test_data[test_start:test_end]
    print(f"Rank {local_rank}: Train chunk {train_start}:{train_end} ({len(train_chunk)} rows), "
          f"Test chunk {test_start}:{test_end} ({len(test_chunk)} rows)")
    if len(train_chunk) == 0 or len(test_chunk) == 0:
        raise ValueError(f"Rank {local_rank}: Empty data chunk (train: {len(train_chunk)}, "
                         f"test: {len(test_chunk)}). Increase data size or reduce world_size.")

    # Initialize rank-specific trading environments
    train_env = TradingEnvironment(train_chunk, initial_balance=100000.0)
    test_env = TradingEnvironment(test_chunk, initial_balance=100000.0)

    # Genetic Algorithm (GA) training
    ga_model_path = "ga_policy_model.pth"
    start_time = time.time()
    if not os.path.exists(ga_model_path):
        ga_agent, best_fit = run_ga_evolution(
            train_env,
            population_size=40,      # Smaller population for faster iteration
            generations=20,          # Limited generations for quick testing
            device=device,
            model_save_path=ga_model_path,
            distributed=True,       # Enable distributed fitness evaluation
            local_rank=local_rank,
            world_size=world_size
        )
        if local_rank == 0:
            print(f"GA best fitness: {best_fit:.2f}")
    dist.barrier()  # Synchronize ranks after GA training

    # Load trained GA model on all ranks
    ga_agent = PolicyNetwork(train_env.observation_dim, 64, train_env.action_space.n, device=device)
    ga_agent.load_model(ga_model_path)
    ga_training_time = time.time() - start_time
    if local_rank == 0:
        print(f"GA training completed in {ga_training_time:.2f} seconds")

    # Evaluate GA agent performance
    balance_history_ga = evaluate_agent_distributed(test_env, ga_agent, local_rank, world_size)
    if local_rank == 0:
        cagr_ga, sharpe_ga, mdd_ga = compute_performance_metrics(balance_history_ga)
        print(f"GA Results - CAGR: {cagr_ga:.4f}, Sharpe: {sharpe_ga:.4f}, MaxDD: {mdd_ga:.4f}")

    # Proximal Policy Optimization (PPO) training
    ppo_model_path = "ppo_actor_critic_model.pth"
    ppo_trainer = PPOTrainer(
        env=train_env,
        input_dim=train_env.observation_space.shape[0],  # Observation space dimension
        action_dim=train_env.action_space.n,             # Number of discrete actions
        hidden_dim=64,                                   # Hidden layer size
        lr=3e-4,                                         # Learning rate
        gamma=0.99,                                      # Discount factor
        gae_lambda=0.95,                                 # GAE smoothing parameter
        clip_epsilon=0.2,                                # PPO clipping parameter
        update_epochs=4,                                 # Epochs per update
        rollout_steps=2048,                              # Steps per rollout
        batch_size=64,                                   # Mini-batch size
        device=device,                                   # GPU device for this rank
        model_save_path=ppo_model_path,
        local_rank=local_rank                            # Rank for distributed coordination
    )
    # Enable distributed training with DDP
    ppo_trainer.model = DDP(ppo_trainer.model, device_ids=[local_rank])

    start_time = time.time()
    if not os.path.exists(ppo_model_path):
        ppo_trainer.train(total_timesteps=1000000 // world_size)  # Scale timesteps by number of GPUs
        if local_rank == 0:
            ppo_trainer.model.module.save_model(ppo_model_path)   # Save model from rank 0 only
    else:
        ppo_trainer.model.module.load_model(ppo_model_path)
        if local_rank == 0:
            print("Loaded existing PPO model.")
    dist.barrier()  # Synchronize ranks after PPO training
    ppo_training_time = time.time() - start_time
    if local_rank == 0:
        print(f"PPO training completed in {ppo_training_time:.2f} seconds")

    # Evaluate PPO agent performance
    balance_history_ppo = evaluate_agent_distributed(test_env, ppo_trainer.model.module, local_rank, world_size)
    if local_rank == 0:
        cagr_ppo, sharpe_ppo, mdd_ppo = compute_performance_metrics(balance_history_ppo)
        print(f"PPO Results - CAGR: {cagr_ppo:.4f}, Sharpe: {sharpe_ppo:.4f}, MaxDD: {mdd_ppo:.4f}")

        # Plot equity curves for comparison (rank 0 only)
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