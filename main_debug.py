# main_debug.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cudf
import time
from data_preprocessing import create_environment_data
from ga_policy_evolution import run_ga_evolution, PolicyNetwork
from policy_gradient_methods import PPOTrainer

# 1) Import your new environment & state
from futures_env import FuturesEnv, TimeSeriesState


def build_states_for_futures_env(df):
    """
    Convert each row of df into a TimeSeriesState.
    Here we define the 'features' that the agent will see.
    For instance, we'll use [Open, High, Low, Close, Volume, return, ma_10].
    We also pick 'date_time' as ts, and 'Close' as price (or you might pick something else).
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
        s = TimeSeriesState(ts=ts, price=price, features=features)
        states.append(s)
    return states


def compute_performance_metrics(balance_history):
    steps_per_year = 252 * 390
    total_return = (balance_history[-1] - balance_history[0]) / balance_history[0] if balance_history[0] != 0 else 0
    years = len(balance_history) / steps_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    returns = np.diff(balance_history) / balance_history[:-1] if len(balance_history) > 1 else [0.0]
    ann_mean = np.mean(returns) * steps_per_year
    ann_std = np.std(returns) * np.sqrt(steps_per_year)
    sharpe = ann_mean / (ann_std + 1e-8)
    running_max = np.maximum.accumulate(balance_history)
    mdd = np.max((running_max - balance_history) / running_max) if len(balance_history) > 0 else 0
    return cagr, sharpe, mdd


def evaluate_agent_distributed(env, agent):
    """
    A single-process (debug) version of agent evaluation.
    Returns the profit history as a list.
    """
    profit_history = []
    obs = env.reset()
    done = False
    device = next(agent.parameters()).device

    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    while not done:
        with torch.no_grad():
            if hasattr(agent, 'act'):
                action = agent.act(obs_tensor)
            else:
                policy_logits, _ = agent.forward(obs_tensor)
                dist_ = torch.distributions.Categorical(logits=policy_logits)
                action = dist_.sample().item()

        obs, reward, done, info = env.step(action)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        profit_history.append(info.get('total_profit', 0.0))

    env.render()
    plt.show()
    return profit_history


def main_debug():
    # Single-process debug mode:
    # Manually pick device (CPU if none available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    local_rank = 0
    world_size = 1

    data_folder = './data_txt'
    cache_folder = './cached_data'
    os.makedirs(cache_folder, exist_ok=True)
    train_data_path = os.path.join(cache_folder, 'train_data.parquet')
    test_data_path = os.path.join(cache_folder, 'test_data.parquet')

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

    train_data = cudf.read_parquet(train_data_path).to_pandas()
    test_data = cudf.read_parquet(test_data_path).to_pandas()
    print(f"Total train data size: {len(train_data)}, Total test data size: {len(test_data)}")

    # In single-process debug, we won't shard data
    train_chunk = train_data
    test_chunk = test_data
    print(f"Train chunk {len(train_chunk)} rows, Test chunk {len(test_chunk)} rows")

    train_states = build_states_for_futures_env(train_chunk)
    test_states = build_states_for_futures_env(test_chunk)

    train_env = FuturesEnv(
        states=train_states,
        value_per_tick=12.5,
        tick_size=0.25,
        fill_probability=1.0,
        execution_cost_per_order=0.0,
        log_dir="./logs/futures_env/train_debug"
    )
    test_env = FuturesEnv(
        states=test_states,
        value_per_tick=12.5,
        tick_size=0.25,
        fill_probability=1.0,
        execution_cost_per_order=0.0,
        log_dir="./logs/futures_env/test_debug"
    )

    ga_model_path = "ga_policy_model.pth"
    start_time = time.time()
    if not os.path.exists(ga_model_path):
        ga_agent, best_fit = run_ga_evolution(
            train_env,
            population_size=40,
            generations=20,
            device=device,
            model_save_path=ga_model_path,
            distributed=False,
            local_rank=local_rank,
            world_size=world_size
        )
        print(f"GA best fitness: {best_fit:.2f}")
    input_dim = int(np.prod(train_env.observation_space.shape))
    ga_agent = PolicyNetwork(input_dim, 64, train_env.action_space.n, device=device)
    ga_agent.load_model(ga_model_path)
    ga_training_time = time.time() - start_time
    print(f"GA training completed in {ga_training_time:.2f} seconds")

    balance_history_ga = evaluate_agent_distributed(test_env, ga_agent)
    if len(balance_history_ga) > 1:
        cagr_ga, sharpe_ga, mdd_ga = compute_performance_metrics(balance_history_ga)
        print(f"GA Results - CAGR: {cagr_ga:.4f}, Sharpe: {sharpe_ga:.4f}, MaxDD: {mdd_ga:.4f}")

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
        update_epochs=4,
        rollout_steps=2048,
        batch_size=64,
        device=device,
        model_save_path=ppo_model_path,
        local_rank=local_rank
    )

    start_time = time.time()
    if not os.path.exists(ppo_model_path):
        ppo_trainer.train(total_timesteps=1000000)
        ppo_trainer.model.save_model(ppo_model_path)
    else:
        ppo_trainer.model.load_model(ppo_model_path)
        print("Loaded existing PPO model.")
    ppo_training_time = time.time() - start_time
    print(f"PPO training completed in {ppo_training_time:.2f} seconds")

    balance_history_ppo = evaluate_agent_distributed(test_env, ppo_trainer.model)
    if len(balance_history_ppo) > 1:
        cagr_ppo, sharpe_ppo, mdd_ppo = compute_performance_metrics(balance_history_ppo)
        print(f"PPO Results - CAGR: {cagr_ppo:.4f}, Sharpe: {sharpe_ppo:.4f}, MaxDD: {mdd_ppo:.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(balance_history_ga, label='GA Policy')
        plt.plot(balance_history_ppo, label='PPO Policy')
        plt.title('Equity Curves Comparison (Debug Mode)')
        plt.xlabel('Time Steps')
        plt.ylabel('Total Profit')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main_debug()
