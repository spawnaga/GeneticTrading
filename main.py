import sys
import os
import time
import warnings
import logging

import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import cudf
    has_cudf = True
except Exception as e:
    warnings.warn(f"cudf import failed ({e}); aliasing pandas as cudf. GPU acceleration disabled.")
    import pandas as pd
    sys.modules['cudf'] = pd
    cudf = pd
    has_cudf = False

from data_preprocessing import create_environment_data
from ga_policy_evolution import run_ga_evolution, PolicyNetwork
from policy_gradient_methods import PPOTrainer, ActorCriticNet
from futures_env import FuturesEnv, TimeSeriesState

def setup_logging(local_rank: int) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger()
    logfile = f"main_rank{local_rank}.log"
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
    logger.info(f"Logging initialized for rank {local_rank} → {logfile}")

def build_states_for_futures_env(df_chunk):
    states = []
    for row in df_chunk.itertuples(index=False):
        ts = row.date_time
        price = float(row.Close)
        features = [
            float(row.Open),
            float(row.High),
            float(row.Low),
            float(row.Close),
            float(row.Volume),
            float(row.return_),
            float(row.ma_10)
        ]
        states.append(TimeSeriesState(ts=ts, price=price, features=features))
    return states

def compute_performance_metrics(balance_history, timestamps):
    if len(balance_history) < 2:
        return 0.0, 0.0, 0.0

    bh = np.array(balance_history, dtype=np.float64)
    total_return = bh[-1] / (bh[0] + 1e-8)
    elapsed = (timestamps[-1] - timestamps[0]).total_seconds()
    years = elapsed / (365.25 * 24 * 3600)
    cagr = total_return ** (1/years) - 1 if years > 0 else 0.0

    rets = np.diff(bh) / (bh[:-1] + 1e-8)
    freq = (timestamps[1] - timestamps[0]).total_seconds()
    annual_factor = (365.25 * 24 * 3600) / freq
    ann_mean = np.mean(rets) * annual_factor
    ann_std  = np.std(rets) * np.sqrt(annual_factor)
    sharpe   = ann_mean / (ann_std + 1e-8)

    peak = np.maximum.accumulate(bh)
    drawdowns = (peak - bh) / (peak + 1e-8)
    mdd = np.max(drawdowns)

    return cagr, sharpe, mdd

def evaluate_agent_distributed(env, agent, local_rank):
    profits, times = [], []
    obs = env.reset()
    done = False

    while not done:
        tensor = torch.tensor(obs, dtype=torch.float32) \
                      .unsqueeze(0) \
                      .to(next(agent.parameters()).device)
        with torch.no_grad():
            action = agent.act(tensor)
        obs, _, done, info = env.step(action)
        profits.append(info.get("total_profit", 0.0))
        times.append(info.get("timestamp", env.states[env.current_index-1].ts))

    if local_rank == 0:
        return profits, times
    else:
        return [], []

def main():
    os.environ["NCCL_TIMEOUT"] = "1800000"
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    setup_logging(local_rank)
    logging.info(f"Rank {local_rank}/{world_size} starting on {device} (has_cudf={has_cudf})")

    data_folder  = "./data_txt"
    cache_folder = "./cached_data"
    os.makedirs(cache_folder, exist_ok=True)
    train_path = os.path.join(cache_folder, "train_data.parquet")
    test_path  = os.path.join(cache_folder, "test_data.parquet")

    if local_rank == 0:
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            train_df, test_df, scaler = create_environment_data(
                data_folder=data_folder,
                max_rows=1000,
                use_gpu=has_cudf,
                cache_folder=cache_folder
            )
            train_df.to_parquet(train_path)
            test_df.to_parquet(test_path)
            logging.info("Data cached to parquet.")
        else:
            logging.info("Parquet cache found; skipping preprocessing.")
    dist.barrier()

    if has_cudf:
        full_train = cudf.read_parquet(train_path)
        full_test  = cudf.read_parquet(test_path)
    else:
        import pandas as pd
        full_train = pd.read_parquet(train_path)
        full_test  = pd.read_parquet(test_path)

    n_train, n_test = len(full_train), len(full_test)
    chunk_train    = n_train // world_size
    chunk_test     = n_test  // world_size

    s_t = local_rank * chunk_train
    e_t = (local_rank+1)*chunk_train if local_rank<world_size-1 else n_train
    s_v = local_rank * chunk_test
    e_v = (local_rank+1)*chunk_test  if local_rank<world_size-1 else n_test

    train_slice = full_train.iloc[s_t:e_t]
    test_slice  = full_test.iloc[s_v:e_v]

    if has_cudf:
        train_pd = train_slice.to_pandas()
        test_pd  = test_slice.to_pandas()
    else:
        train_pd = train_slice.copy()
        test_pd  = test_slice.copy()

    train_pd.rename(columns={"return": "return_"}, inplace=True)
    test_pd.rename(columns={"return": "return_"},  inplace=True)

    train_states = build_states_for_futures_env(train_pd)
    test_states  = build_states_for_futures_env(test_pd)

    env_kwargs = {
        "value_per_tick": 12.5,
        "tick_size": 0.25,
        "fill_probability": 1.0,
        "execution_cost_per_order": 0.0005,
        "contracts_per_trade": 1,
        "margin_rate": 0.01,
        "bid_ask_spread": 0.05,
        "add_current_position_to_state": True
    }
    train_env = FuturesEnv(states=train_states,
                           log_dir=f"./logs/train_rank{local_rank}",
                           **env_kwargs)
    env_kwargs["execution_cost_per_order"] = 0.00005
    test_env  = FuturesEnv(states=test_states,
                           log_dir=f"./logs/test_rank{local_rank}",
                           **env_kwargs)

    ga_model = "ga_policy_model.pth"
    if local_rank == 0:
        best_agent, best_fit, _, _ = run_ga_evolution(
            train_env,
            population_size=80,
            generations=100,
            tournament_size=7,
            mutation_rate=0.8,
            mutation_scale=1.0,
            num_workers=2,
            device=str(device),
            model_save_path=ga_model
        )
        logging.info(f"GA training complete – best fitness: {best_fit:.2f}")
        best_agent.save_model(ga_model)
    dist.barrier()

    ga_input_dim = int(np.prod(train_env.observation_space.shape))
    ga_agent = PolicyNetwork(
        input_dim=ga_input_dim,
        hidden_dim=64,
        output_dim=train_env.action_space.n,
        device=str(device)
    )
    ga_agent.load_model(ga_model)
    ga_profits, ga_times = evaluate_agent_distributed(test_env, ga_agent, local_rank)
    if local_rank == 0:
        cagr, sharpe, mdd = compute_performance_metrics(ga_profits, ga_times)
        logging.info(f"GA Eval → CAGR: {cagr:.4f}, Sharpe: {sharpe:.4f}, MDD: {mdd:.4f}")

    ppo_model = "ppo_model.pth"
    ppo_trainer = PPOTrainer(
        env=train_env,
        input_dim=int(np.prod(train_env.observation_space.shape)),
        action_dim=train_env.action_space.n,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        update_epochs=10,
        rollout_steps=512,
        batch_size=64,
        device=str(device),
        model_save_path=ppo_model,
        local_rank=local_rank
    )

    logging.info("About to wrap PPO model in DDP – calling dist.barrier()")
    dist.barrier()
    ppo_trainer.model = DDP(ppo_trainer.model, device_ids=[local_rank])

    if local_rank == 0:
        ppo_trainer.train(total_timesteps=1_000_000 // world_size)
    dist.barrier()

    ppo_agent = ActorCriticNet(
        input_dim=int(np.prod(train_env.observation_space.shape)),
        hidden_dim=64,
        action_dim=train_env.action_space.n,
        device=str(device)
    )
    ppo_agent.load_model(ppo_model)
    ppo_profits, ppo_times = evaluate_agent_distributed(test_env, ppo_agent, local_rank)
    if local_rank == 0:
        cagr, sharpe, mdd = compute_performance_metrics(ppo_profits, ppo_times)
        logging.info(f"PPO Eval → CAGR: {cagr:.4f}, Sharpe: {sharpe:.4f}, MDD: {mdd:.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()