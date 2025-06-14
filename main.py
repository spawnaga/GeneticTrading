#!/usr/bin/env python
import os
# ─── SILENCE TensorFlow / oneDNN / XLA CUDA logs ───────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]       = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]      = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# ────────────────────────────────────────────────────────────────────────────────

import logging
import time
import sys
import warnings
import pickle
import collections

import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

# Try GPU DataFrame support
try:
    import cudf
    import cupy as cp
    has_cudf = True
except Exception as e:
    warnings.warn(f"cudf import failed ({e}); aliasing pandas as cudf. GPU acceleration disabled.")
    import pandas as pd
    sys.modules['cudf'] = pd
    cudf = pd
    has_cudf = False

# Local imports
from data_preprocessing import create_environment_data
from ga_policy_evolution import run_ga_evolution, PolicyNetwork
from policy_gradient_methods import PPOTrainer, ActorCriticNet
from utils import evaluate_agent_distributed, compute_performance_metrics
from futures_env import FuturesEnv, TimeSeriesState

# ──────────────────────────────────────────────────────────────────────────────
# Globals for live‐inference buffering
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_MA      = 10
WINDOW_RSI     = 14
WINDOW_VOL     = 10
BIN_SIZE       = 15
SECONDS_IN_DAY = 24 * 60

history = {
    "closes":  collections.deque(maxlen=WINDOW_MA),
    "deltas":  collections.deque(maxlen=WINDOW_RSI),
    "returns": collections.deque(maxlen=WINDOW_VOL),
}

scaler: "cuml.preprocessing.StandardScaler" = None
segment_dict: dict[int,int] = {}

# ──────────────────────────────────────────────────────────────────────────────
def setup_logging(local_rank: int) -> None:
    """
    Configure root logger to write to both console and a per‐rank log file.
    """
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger()
    logfile = f"main_rank{local_rank}.log"
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
    logger.info(f"Logging initialized for rank {local_rank} → {logfile}")

# ──────────────────────────────────────────────────────────────────────────────
def build_states_for_futures_env(df_chunk):
    """
    Convert a DataFrame slice into a list of TimeSeriesState for FuturesEnv.
    """
    states = []
    for row in df_chunk.itertuples(index=False):
        feats = [
            row.Open, row.High, row.Low, row.Close,
            row.Volume, row.return_, row.ma_10,
            row.rsi, row.volatility,
            row.sin_time, row.cos_time,
            row.sin_weekday, row.cos_weekday,
        ]
        for col in df_chunk.columns:
            if col.startswith("tb_") or col.startswith("wd_"):
                feats.append(getattr(row, col))
        states.append(
            TimeSeriesState(
                ts=row.date_time,
                open_price=row.Open,
                close_price=row.Close,
                features=feats,
            )
        )
    return states

# ──────────────────────────────────────────────────────────────────────────────
def compute_performance_metrics(balance_history, timestamps):
    """
    Compute CAGR, Sharpe ratio, and maximum drawdown from profit history.
    """
    if len(balance_history) < 2:
        return 0.0, 0.0, 0.0

    bh      = np.array(balance_history, dtype=np.float64)
    total   = bh[-1] / (bh[0] + 1e-8)
    elapsed = (timestamps[-1] - timestamps[0]).total_seconds()
    years   = elapsed / (365.25 * 24 * 3600)
    cagr    = total**(1/years) - 1 if years > 0 else 0.0

    rets    = np.diff(bh) / (bh[:-1] + 1e-8)
    freq    = (timestamps[1] - timestamps[0]).total_seconds()
    ann_fac = (365.25 * 24 * 3600) / freq
    ann_mean = np.mean(rets) * ann_fac
    ann_std  = np.std(rets) * np.sqrt(ann_fac)
    sharpe   = ann_mean / (ann_std + 1e-8)

    peak       = np.maximum.accumulate(bh)
    drawdowns  = (peak - bh) / (peak + 1e-8)
    mdd        = np.max(drawdowns)

    return cagr, sharpe, mdd

# ──────────────────────────────────────────────────────────────────────────────
def evaluate_agent_distributed(env, agent, local_rank):
    """
    Run the trained agent in the test environment. Only rank 0 returns results.
    """
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

    return (profits, times) if local_rank == 0 else ([], [])

# ──────────────────────────────────────────────────────────────────────────────
def process_live_row(bar: dict) -> "cudf.DataFrame":
    """
    Given a single OHLCV bar dict, compute all features exactly as in training,
    apply the saved scaler, and return a 1-row cuDF DataFrame ready for inference.
    """
    global history, scaler, segment_dict
    ts    = bar["date_time"]
    close = bar["Close"]
    history["closes"].append(close)

    if len(history["closes"]) >= 2:
        prev  = history["closes"][-2]
        delta = close - prev
        rtn   = delta / (prev + 1e-8)
    else:
        delta, rtn = 0.0, 0.0

    history["deltas"].append(delta)
    history["returns"].append(rtn)

    ma10  = float(np.mean(history["closes"]))
    gain  = float(np.mean([d for d in history["deltas"] if d > 0])) if history["deltas"] else 0.0
    loss  = float(np.mean([-d for d in history["deltas"] if d < 0])) if history["deltas"] else 1e-8
    rs    = gain / (loss + 1e-8)
    rsi   = 100 - (100 / (1 + rs))
    vol   = float(np.std(history["returns"])) if len(history["returns"]) > 1 else 0.0

    weekday = ts.weekday()
    minutes = ts.hour * 60 + ts.minute
    theta_t = 2 * np.pi * (minutes / SECONDS_IN_DAY)
    theta_w = 2 * np.pi * (weekday / 7)
    sin_t, cos_t = np.sin(theta_t), np.cos(theta_t)
    sin_w, cos_w = np.sin(theta_w), np.cos(theta_w)

    bin_index = minutes // BIN_SIZE
    ohe = {}
    for d in range(7):
        ohe[f"wd_{d}"] = 1 if d == weekday else 0
    total_bins = SECONDS_IN_DAY // BIN_SIZE
    for b in range(total_bins):
        ohe[f"tb_{b}"] = 1 if b == bin_index else 0

    row = {
        "Open": bar["Open"],   "High": bar["High"],
        "Low": bar["Low"],     "Close": close,
        "Volume": bar["Volume"], "return": rtn,
        "ma_10": ma10,         "rsi": rsi,
        "volatility": vol,     "sin_time": sin_t,
        "cos_time": cos_t,     "sin_weekday": sin_w,
        "cos_weekday": cos_w,
        **ohe
    }

    df_live = cudf.DataFrame([row])
    num_cols = ["Open","High","Low","Close","Volume","return","ma_10"]
    df_live[num_cols] = scaler.transform(df_live[num_cols])
    return df_live

# ──────────────────────────────────────────────────────────────────────────────
def main():
    # suppress TF logs further
    os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["NCCL_TIMEOUT"]         = "1800000"

    # torchrun / torch.distributed sets LOCAL_RANK
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()

    setup_logging(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    logging.info(f"Rank {local_rank}/{world_size} starting on {device} (has_cudf={has_cudf})")

    # ─── Data prep ────────────────────────────────────────────────────────────────
    data_folder  = "./data_txt"
    cache_folder = "./cached_data"
    os.makedirs(cache_folder, exist_ok=True)
    train_path = os.path.join(cache_folder, "train_data.parquet")
    test_path  = os.path.join(cache_folder, "test_data.parquet")

    if local_rank == 0:
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            train_df, test_df, scaler, segment_dict = create_environment_data(
                data_folder=data_folder,
                max_rows=1000,
                cache_folder=cache_folder,
                use_gpu=has_cudf,
                test_size=0.2
            )
            train_df.to_parquet(train_path, index=False)
            test_df.to_parquet(test_path,  index=False)
            with open(os.path.join(cache_folder, "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            with open(os.path.join(cache_folder, "segment_dict.pkl"), "wb") as f:
                pickle.dump(segment_dict, f)
            logging.info("Data cached to parquet and artifacts saved.")
        else:
            logging.info("Parquet cache found; skipping preprocessing.")
    dist.barrier(device_ids=[local_rank])

    if has_cudf:
        full_train = cudf.read_parquet(train_path)
        full_test  = cudf.read_parquet(test_path)
    else:
        import pandas as pd
        full_train = pd.read_parquet(train_path)
        full_test  = pd.read_parquet(test_path)

    with open(os.path.join(cache_folder, "scaler.pkl"),     "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(cache_folder, "segment_dict.pkl"), "rb") as f:
        segment_dict = pickle.load(f)

    # ─── Shard for DDP ────────────────────────────────────────────────────────────
    n_train, n_test = len(full_train), len(full_test)
    per_rank_t = n_train // world_size
    per_rank_v = n_test  // world_size
    s_t = local_rank * per_rank_t
    e_t = s_t + per_rank_t if local_rank < world_size-1 else n_train
    s_v = local_rank * per_rank_v
    e_v = s_v + per_rank_v if local_rank < world_size-1 else n_test

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

    base_env_kwargs = {
        "value_per_tick": 12.5,
        "tick_size": 0.25,
        "fill_probability": 1.0,
        "execution_cost_per_order": 0.0005,
        "contracts_per_trade": 1,
        "margin_rate": 0.01,
        "bid_ask_spread": 0.25,
        "add_current_position_to_state": True
    }
    train_env = FuturesEnv(states=train_states,
                           log_dir=f"./logs/train_rank{local_rank}",
                           **base_env_kwargs)
    base_env_kwargs["execution_cost_per_order"] = 0.00005
    test_env  = FuturesEnv(states=test_states,
                           log_dir=f"./logs/test_rank{local_rank}",
                           **base_env_kwargs)

    # ─── GA Training / Eval ─────────────────────────────────────────────────────
    ga_model = "ga_policy_model.pth"
    if local_rank == 0 and not os.path.exists(ga_model):
        best_agent, best_fit, _, _ = run_ga_evolution(
            train_env,
            population_size=80,
            generations=100,
            tournament_size=7,
            mutation_rate=0.8,
            mutation_scale=1.0,
            num_workers=4,
            device=str(device),
            model_save_path=ga_model
        )
        logging.info(f"GA training complete – best fitness: {best_fit:.2f}")
        best_agent.save_model(ga_model)
    dist.barrier(device_ids=[local_rank])

    ga_agent = PolicyNetwork(
        input_dim=int(np.prod(train_env.observation_space.shape)),
        hidden_dim=64,
        output_dim=train_env.action_space.n,
        device=str(device)
    )
    ga_agent.load_model(ga_model)
    ga_profits, ga_times = evaluate_agent_distributed(test_env, ga_agent, local_rank)
    if local_rank == 0:
        cagr, sharpe, mdd = compute_performance_metrics(ga_profits, ga_times)
        logging.info(f"GA Eval → CAGR: {cagr:.4f}, Sharpe: {sharpe:.4f}, MDD: {mdd:.4f}")

    # ─── PPO Training / Eval ─────────────────────────────────────────────────────
    ppo_model = "ppo_model.pth"
    # total timesteps split across ranks
    total_timesteps = 1_000_000
    per_rank_steps  = total_timesteps // world_size

    # build trainer with local_rank so it knows who shows the bar
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
        local_rank=local_rank,
        eval_interval=10
    )

    # — Load / resume before wrapping in DDP
    ckpt_path = ppo_model + ".ckpt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ppo_trainer.model.load_state_dict(ckpt["model_state"])
        ppo_trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        ppo_trainer.scheduler.load_state_dict(ckpt["scheduler_state"])
        ppo_trainer.entropy_coef = ckpt["entropy_coef"]
        last_upd = ckpt.get("update_idx", -1)
        max_upd  = max(1, per_rank_steps // ppo_trainer.rollout_steps)
        start_update = last_upd + 1 if last_upd + 1 < max_upd else 0
        logging.info(f"Resuming PPO from update {start_update}")
    else:
        start_update = 0
        logging.info("No PPO checkpoint found; starting from scratch")

    # — Wrap in DDP and barrier
    logging.info("Wrapping PPO model in DDP – waiting at barrier")
    dist.barrier(device_ids=[local_rank])
    ppo_trainer.model = DDP(ppo_trainer.model, device_ids=[local_rank])

    # — Run training
    ppo_trainer.train(
        total_timesteps=per_rank_steps,
        start_update=start_update,
        eval_env=test_env
    )
    dist.barrier(device_ids=[local_rank])

    # — Final evaluation
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
