import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import os
import cudf
import time
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP

from data_preprocessing import create_environment_data
from ga_policy_evolution import run_ga_evolution, PolicyNetwork
from policy_gradient_methods import PPOTrainer

# 1) Import your new environment & state
from futures_env import FuturesEnv, TimeSeriesState


def build_states_for_futures_env(df):
    """
    Convert each row of a pandas DataFrame into a TimeSeriesState object.
    Each state comprises a timestamp (ts), a price, and a set of numeric features.

    For example, we use the columns:
      - date_time (converted to ts)
      - Close (price)
      - [Open, High, Low, Close, Volume, return, ma_10] as the features.

    Feel free to adjust these features to match your data's structure.
    """
    states = []
    for i, row in df.iterrows():
        ts = row['date_time']
        price = float(row['Close'])
        features = [
            row['Open'], row['High'], row['Low'],
            row['Close'], row['Volume'], row['return'],
            row['ma_10']
        ]
        # Convert each row into a TimeSeriesState
        s = TimeSeriesState(ts=ts, price=price, features=features)
        states.append(s)
    return states


def compute_performance_metrics(balance_history):
    """
    Compute standard performance metrics:
      - CAGR
      - Sharpe Ratio
      - Maximum Drawdown
    from a list/array of running PnL or 'balance' values (balance_history).
    """
    steps_per_year = 252 * 390  # approximate trading days x minutes
    if len(balance_history) <= 1:
        return 0.0, 0.0, 0.0

    # Defensive check to avoid division by zero if the first element is 0
    if balance_history[0] == 0:
        total_return = 0
    else:
        total_return = (balance_history[-1] - balance_history[0]) / balance_history[0]

    years = len(balance_history) / steps_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    returns = np.diff(balance_history) / np.array(balance_history[:-1], dtype=np.float64)
    ann_mean = np.mean(returns) * steps_per_year
    ann_std = np.std(returns) * np.sqrt(steps_per_year)
    sharpe = ann_mean / (ann_std + 1e-8)

    running_max = np.maximum.accumulate(balance_history)
    if len(running_max) > 0:
        mdd = np.max((running_max - balance_history) / running_max)
    else:
        mdd = 0.0

    return cagr, sharpe, mdd


def evaluate_agent_distributed(env, agent, local_rank, world_size):
    """
    Evaluate an agent's performance across all distributed ranks on the given environment.

    Returns a profit_history list only on rank 0 (others get local partial lists).
    The environment's built-in render() method is called on rank 0
    to produce any histograms or distribution plots of trades.
    """
    profit_history = []

    obs = env.reset()
    done = False
    device = next(agent.parameters()).device

    # Convert obs to Torch tensor
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    while not done:
        with torch.no_grad():
            if hasattr(agent, 'act'):
                # GA or Q-like policy: agent.act(...) picks the best action
                action = agent.act(obs_tensor)
            else:
                # PPO actor: forward -> sample an action from the policy distribution
                policy_logits, _ = agent.forward(obs_tensor)
                dist_ = torch.distributions.Categorical(logits=policy_logits)
                action = dist_.sample().item()

        obs, reward, done, info = env.step(action)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        # Info dict has 'total_profit'
        profit_history.append(info.get('total_profit', 0.0))

    # Gather profit histories from each rank to rank 0
    profit_tensor = torch.tensor(profit_history, dtype=torch.float32).to(device)
    gather_list = (
        [torch.zeros(len(profit_history), dtype=torch.float32).to(device) for _ in range(world_size)]
        if local_rank == 0 else None
    )
    dist.gather(
        profit_tensor,
        gather_list=gather_list if local_rank == 0 else None,
        dst=0
    )

    # On rank 0, we can combine them and produce final results
    if local_rank == 0:
        full_profit_history = torch.cat(gather_list).cpu().numpy().tolist()
        # Render environment charts (histograms, distributions of trades, etc.)
        env.render()
        plt.show()
        return full_profit_history

    return profit_history


def main():
    """
    Main function for distributed training & testing with GA and PPO on a futures environment.

    Steps:
      1) Distributed init, environment creation, data loading & sharding
      2) GA training, evaluation
      3) PPO training, evaluation
      4) Single unbroken backtest for GA
      5) Single unbroken backtest for PPO
      6) Plot & save final trades for both.

    Produces 4 CSV files:
      final_ga_orders.csv, final_ga_trades.csv
      final_ppo_orders.csv, final_ppo_trades.csv
    And 2 single-run backtest plots for GA & PPO.
    """
    # ------------------------------------------------
    # 1) Initialize distributed environment
    # ------------------------------------------------
    timeout = datetime.timedelta(seconds=3600)
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timeout)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ------------------------------------------------
    # 2) Data loading / caching
    # ------------------------------------------------
    data_folder = './data_txt'
    cache_folder = './cached_data'
    os.makedirs(cache_folder, exist_ok=True)
    train_data_path = os.path.join(cache_folder, 'train_data.parquet')
    test_data_path = os.path.join(cache_folder, 'test_data.parquet')

    # --------------------------------------------------------------
    # Data preparation (rank 0 only: loads & caches data to parquet)
    # --------------------------------------------------------------
    if local_rank == 0:
        if not (os.path.exists(train_data_path) and os.path.exists(test_data_path)):
            train_data, test_data, scaler = create_environment_data(
                data_folder=data_folder,
                max_rows=1000,
                use_gpu=True,
                cache_folder=cache_folder
            )
            train_data.to_parquet(train_data_path)
            test_data.to_parquet(test_data_path)
            print("✅ Data cached successfully.")
        else:
            print("✅ Cached data already exists.")
    dist.barrier()  # Sync all ranks

    # Load data on all ranks from cached parquet
    train_data = cudf.read_parquet(train_data_path).to_pandas()
    test_data = cudf.read_parquet(test_data_path).to_pandas()

    if local_rank == 0:
        print(f"Total train data size: {len(train_data)}, Total test data size: {len(test_data)}")

    # Shard data across ranks to ensure balanced workloads
    train_chunk_size = max(1, len(train_data) // world_size)
    test_chunk_size = max(1, len(test_data) // world_size)
    train_start = local_rank * train_chunk_size
    train_end = min((local_rank + 1) * train_chunk_size, len(train_data))
    test_start = local_rank * test_chunk_size
    test_end = min((local_rank + 1) * test_chunk_size, len(test_data))

    train_chunk = train_data[train_start:train_end]
    test_chunk = test_data[test_start:test_end]

    print(
        f"Rank {local_rank}: Train chunk {train_start}:{train_end} "
        f"({len(train_chunk)} rows), Test chunk {test_start}:{test_end} "
        f"({len(test_chunk)} rows)"
    )
    if len(train_chunk) == 0 or len(test_chunk) == 0:
        raise ValueError(f"Rank {local_rank}: Empty data chunk.")

    # Convert shards to TimeSeriesState sequences
    train_states = build_states_for_futures_env(train_chunk)
    test_states = build_states_for_futures_env(test_chunk)

    # ------------------------------------------------
    # 3) Create environment
    # ------------------------------------------------
    train_env = FuturesEnv(
        states=train_states,
        value_per_tick=12.5,
        tick_size=0.25,
        fill_probability=1.0,
        execution_cost_per_order=0.0,
        log_dir=f"./logs/futures_env/train_rank{local_rank}"
    )
    test_env = FuturesEnv(
        states=test_states,
        value_per_tick=12.5,
        tick_size=0.25,
        fill_probability=1.0,
        execution_cost_per_order=0.0,
        log_dir=f"./logs/futures_env/test_rank{local_rank}"
    )

    # ------------------------------------------------
    # 4) GA (Genetic Algorithm) training
    # ------------------------------------------------
    ga_model_path = "ga_policy_model.pth"
    start_time = time.time()
    if not os.path.exists(ga_model_path):
        ga_agent, best_fit = run_ga_evolution(
            train_env,
            population_size=40,
            generations=20,
            device=device,
            model_save_path=ga_model_path,
            distributed=True,
            local_rank=local_rank,
            world_size=world_size
        )
        if local_rank == 0:
            print(f"GA best fitness: {best_fit:.2f}")
    dist.barrier()

    # Load GA agent
    input_dim = int(np.prod(train_env.observation_space.shape))
    ga_agent = PolicyNetwork(input_dim, 64, train_env.action_space.n, device=device)
    ga_agent.load_model(ga_model_path)
    ga_training_time = time.time() - start_time
    if local_rank == 0:
        print(f"GA training completed in {ga_training_time:.2f} seconds")

    # Evaluate GA on test_env
    balance_history_ga = evaluate_agent_distributed(test_env, ga_agent, local_rank, world_size)
    if local_rank == 0:
        if len(balance_history_ga) > 1:
            cagr_ga, sharpe_ga, mdd_ga = compute_performance_metrics(balance_history_ga)
            print(f"GA Results - CAGR: {cagr_ga:.4f}, Sharpe: {sharpe_ga:.4f}, MaxDD: {mdd_ga:.4f}")

    # ------------------------------------------------
    # 5) PPO training
    # ------------------------------------------------
    ppo_model_path = "ppo_actor_critic_model.pth"

    ppo_trainer = PPOTrainer(
        env=train_env,
        input_dim=input_dim,
        action_dim=train_env.action_space.n,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        update_epochs=10,  # increased from default 4 to help training
        rollout_steps=512,  # shortened from 2048 for more frequent updates
        batch_size=64,
        device=device,
        model_save_path=ppo_model_path,
        local_rank=local_rank
    )
    ppo_trainer.model = DDP(ppo_trainer.model, device_ids=[local_rank])

    start_time = time.time()
    if not os.path.exists(ppo_model_path):
        # Train PPO
        ppo_trainer.train(total_timesteps=1000000 // world_size)
        if local_rank == 0:
            ppo_trainer.model.module.save_model(ppo_model_path)
    else:
        ppo_trainer.model.module.load_model(ppo_model_path)
        if local_rank == 0:
            print("Loaded existing PPO model.")
    dist.barrier()

    ppo_training_time = time.time() - start_time
    if local_rank == 0:
        print(f"PPO training completed in {ppo_training_time:.2f} seconds")

    # Evaluate PPO
    balance_history_ppo = evaluate_agent_distributed(test_env, ppo_trainer.model.module, local_rank, world_size)

    # ------------------------------------------------
    # 6) SINGLE UNBROKEN BACKTEST for GA
    #    (to avoid "sharp falling line" from multiple partial runs)
    # ------------------------------------------------
    if local_rank == 0:
        # GA single-run final backtest
        best_ga_agent = PolicyNetwork(input_dim, 64, train_env.action_space.n, device=device)
        best_ga_agent.load_model(ga_model_path)

        # We'll run one single test from start to finish on the test set
        final_env = FuturesEnv(
            states=test_states,  # or train_states if you prefer
            value_per_tick=12.5,
            tick_size=0.25,
            fill_probability=1.0,
            execution_cost_per_order=0.0,
            log_dir="./logs/final_ga_test"
        )

        obs = final_env.reset()
        done = False
        profit_history = []
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = best_ga_agent.act(obs_tensor)
            obs, reward, done, info = final_env.step(action)
            profit_history.append(info.get("total_profit", 0.0))

        # Optionally, remove the final data point if it causes a drop you don't want to see:
        # if len(profit_history) > 0:
        #     profit_history.pop()

        # Save orders & trades to CSV for debugging or inspection
        final_orders_df, final_trades_df = final_env.get_episode_data()
        final_orders_df.to_csv("final_ga_orders.csv", index=False)
        final_trades_df.to_csv("final_ga_trades.csv", index=False)
        print("Saved final GA orders & trades to CSV.")

        # Plot final GA single-run equity curve
        plt.figure()
        plt.plot(profit_history, label="GA Final Single Run")
        plt.title("GA Single Unbroken Backtest")
        plt.xlabel("Steps")
        plt.ylabel("Total Profit")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ------------------------------------------------
    # 7) SINGLE UNBROKEN BACKTEST for PPO
    #    (to produce distinct CSV and plot)
    # ------------------------------------------------

    # if local_rank == 0:
    #     # For PPO, use the ActorCriticNet since the PPO model was trained with it.
    #     best_ppo_net = ActorCriticNet(
    #         input_dim=input_dim,
    #         hidden_dim=64,
    #         action_dim=train_env.action_space.n,
    #         device=device
    #     )
    #     best_ppo_net.load_model(ppo_model_path)
    #
    #     final_ppo_env = FuturesEnv(
    #         states=test_states,
    #         value_per_tick=12.5,
    #         tick_size=0.25,
    #         fill_probability=1.0,
    #         execution_cost_per_order=0.0,
    #         log_dir="./logs/final_ppo_test"
    #     )
    #
    #     obs = final_ppo_env.reset()
    #     done = False
    #     final_ppo_profit_history = []
    #     while not done:
    #         obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    #         with torch.no_grad():
    #             policy_logits, _ = best_ppo_net(obs_tensor)
    #             dist_ = torch.distributions.Categorical(logits=policy_logits)
    #             action = dist_.sample().item()
    #         obs, reward, done, info = final_ppo_env.step(action)
    #         final_ppo_profit_history.append(info.get("total_profit", 0.0))
    #
    #     # Save final PPO orders & trades to CSV for later inspection
    #     ppo_orders_df, ppo_trades_df = final_ppo_env.get_episode_data()
    #     ppo_orders_df.to_csv("final_ppo_orders.csv", index=False)
    #     ppo_trades_df.to_csv("final_ppo_trades.csv", index=False)
    #     print("Saved final PPO orders & trades to CSV.")
    #
    #     # Plot final PPO single-run equity curve
    #     plt.figure()
    #     plt.plot(final_ppo_profit_history, label="PPO Final Single Run")
    #     plt.title("PPO Single Unbroken Backtest")
    #     plt.xlabel("Steps")
    #     plt.ylabel("Total Profit")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    # ------------------------------------------------
    # Optionally, if you want to see combined final curves for GA/PPO:
    #    (currently commented out but retained for reference)
    # ------------------------------------------------
    # if local_rank == 0:
    #     if len(balance_history_ppo) > 1:
    #         cagr_ppo, sharpe_ppo, mdd_ppo = compute_performance_metrics(balance_history_ppo)
    #         print(f"PPO Results - CAGR: {cagr_ppo:.4f}, Sharpe: {sharpe_ppo:.4f}, MaxDD: {mdd_ppo:.4f}")
    #
    #         # Plot equity curves for entire test run (not the single-run final backtest)
    #         plt.figure(figsize=(10, 6))
    #         plt.plot(balance_history_ga, label='GA Policy (test run)')
    #         plt.plot(balance_history_ppo, label='PPO Policy (test run)')
    #         plt.title('Equity Curves Comparison')
    #         plt.xlabel('Time Steps')
    #         plt.ylabel('Total Profit')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()

    # Clean up distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
