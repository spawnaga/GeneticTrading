#!/usr/bin/env python
import os
# ─── SILENCE TensorFlow / oneDNN / XLA CUDA logs ───────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"   # only error messages
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"   # disable oneDNN logging
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# ────────────────────────────────────────────────────────────────────────────────

import sys
import time
import warnings
import logging
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
# Globals for live-inference buffering
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_MA      = 10
WINDOW_RSI     = 14
WINDOW_VOL     = 10
BIN_SIZE       = 15
SECONDS_IN_DAY = 24 * 60

# Buffers to compute rolling features on live data
history = {
    "closes":  collections.deque(maxlen=WINDOW_MA),
    "deltas":  collections.deque(maxlen=WINDOW_RSI),
    "returns": collections.deque(maxlen=WINDOW_VOL),
}

# Artifacts loaded once at startup
scaler: "cuml.preprocessing.StandardScaler" = None
segment_dict: dict[int,int] = {}

# ──────────────────────────────────────────────────────────────────────────────
def setup_logging(local_rank: int) -> None:
    """
    Configure root logger to write to both console and a per-rank log file.
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
            # plus all tb_* and wd_* one-hots…
        ]
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

    # 1) Update buffers & compute raw features
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

    # 2) Time features
    weekday = ts.weekday()
    minutes = ts.hour * 60 + ts.minute
    theta_t = 2 * np.pi * (minutes / SECONDS_IN_DAY)
    theta_w = 2 * np.pi * (weekday / 7)
    sin_t, cos_t = np.sin(theta_t), np.cos(theta_t)
    sin_w, cos_w = np.sin(theta_w), np.cos(theta_w)

    # 3) One-hot via segment_dict mapping
    bin_index = minutes // BIN_SIZE
    seg_key   = weekday * (SECONDS_IN_DAY // BIN_SIZE) + bin_index
    # Build a zeroed dictionary for all OHE cols
    ohe = {}
    # weekday one-hot
    for d in range(7):
        ohe[f"wd_{d}"] = 1 if d == weekday else 0
    # time-bin one-hot
    total_bins = SECONDS_IN_DAY // BIN_SIZE
    for b in range(total_bins):
        ohe[f"tb_{b}"] = 1 if b == bin_index else 0

    # 4) Assemble row dict
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

    # 5) Build cuDF, scale numeric columns
    df_live = cudf.DataFrame([row])
    num_cols = ["Open","High","Low","Close","Volume","return","ma_10"]
    df_live[num_cols] = scaler.transform(df_live[num_cols])

    return df_live

# ──────────────────────────────────────────────────────────────────────────────
def main():

    # suppress TF logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["NCCL_TIMEOUT"]         = "1800000"

    # 1) Pull in the torchrun‐provided env vars
    local_rank = int(os.environ["LOCAL_RANK"])
    # (RANK and WORLD_SIZE are also set if you need them explicitly)
    # 2) Tell PyTorch which GPU this process should use
    torch.cuda.set_device(local_rank)

    # 3) Initialize the default process group
    dist.init_process_group(backend="nccl", init_method="env://")

    # Now you can safely call:
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    setup_logging(local_rank)
    logging.info(f"Rank {rank}/{world_size} starting on cuda:{local_rank} (has_cudf={has_cudf})")

    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    setup_logging(local_rank)
    logging.info(f"Rank {local_rank}/{world_size} starting on {device} (has_cudf={has_cudf})")

    # Load or preprocess train/test data once (rank 0), then barrier
    data_folder  = "./data_txt"
    cache_folder = "./cached_data"
    os.makedirs(cache_folder, exist_ok=True)
    train_path = os.path.join(cache_folder, "train_data.parquet")
    test_path  = os.path.join(cache_folder, "test_data.parquet")

    if local_rank == 0:
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            # NOTE: unpack the four returns: train, test, scaler, segment_dict
            train_df, test_df, scaler, segment_dict = create_environment_data(
                data_folder=data_folder,
                max_rows=1000,
                cache_folder=cache_folder,
                use_gpu=has_cudf,
                test_size=0.2
            )
            # For pandas-backed DataFrames you can still pass index=False; cudf ignores it
            train_df.to_parquet(train_path, index=False)
            test_df.to_parquet(test_path,  index=False)
            # Persist scaler & segment_dict for live inference
            with open(os.path.join(cache_folder, "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
            with open(os.path.join(cache_folder, "segment_dict.pkl"), "wb") as f:
                pickle.dump(segment_dict, f)
            logging.info("Data cached to parquet and artifacts saved.")
        else:
            logging.info("Parquet cache found; skipping preprocessing.")
    dist.barrier(device_ids=[local_rank])

    # Load preprocessed data
    if has_cudf:
        full_train = cudf.read_parquet(train_path)
        full_test  = cudf.read_parquet(test_path)
    else:
        import pandas as pd
        full_train = pd.read_parquet(train_path)
        full_test  = pd.read_parquet(test_path)

    # Load scaler & segment_dict for live inference
    with open(os.path.join(cache_folder, "scaler.pkl"),     "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(cache_folder, "segment_dict.pkl"), "rb") as f:
        segment_dict = pickle.load(f)

    # Split data across ranks
    n_train = len(full_train); n_test = len(full_test)
    chunk_t = n_train // world_size
    chunk_v = n_test  // world_size
    s_t = local_rank * chunk_t
    e_t = s_t + chunk_t if local_rank < world_size-1 else n_train
    s_v = local_rank * chunk_v
    e_v = s_v + chunk_v if local_rank < world_size-1 else n_test

    train_slice = full_train.iloc[s_t:e_t]
    test_slice  = full_test.iloc[s_v:e_v]

    # Convert to pandas for env (TimeSeriesState expects pandas types)
    if has_cudf:
        train_pd = train_slice.to_pandas()
        test_pd  = test_slice.to_pandas()
    else:
        train_pd = train_slice.copy()
        test_pd  = test_slice.copy()

    # Rename return column to avoid keyword clash
    train_pd.rename(columns={"return": "return_"}, inplace=True)
    test_pd.rename(columns={"return": "return_"},  inplace=True)

    # Build environment states
    train_states = build_states_for_futures_env(train_pd)
    test_states  = build_states_for_futures_env(test_pd)

    # Common env kwargs
    env_kwargs = {
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
                           **env_kwargs)
    env_kwargs["execution_cost_per_order"] = 0.00005
    test_env  = FuturesEnv(states=test_states,
                           log_dir=f"./logs/test_rank{local_rank}",
                           **env_kwargs)

    # ──────────────────────────────────────────────────────────────────────────
    # Genetic Algorithm Training & Evaluation
    # ──────────────────────────────────────────────────────────────────────────
    ga_model = "ga_policy_model.pth"
    if local_rank == 0:
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

    # ──────────────────────────────────────────────────────────────────────────
    # PPO Training & Evaluation
    # ──────────────────────────────────────────────────────────────────────────
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

    logging.info("Wrapping PPO model in DDP – waiting at barrier")
    dist.barrier(device_ids=[local_rank])
    ppo_trainer.model = DDP(ppo_trainer.model, device_ids=[local_rank])

    # 1) Load checkpoint (if any) to resume
    ckpt_path = ppo_model + ".ckpt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ppo_trainer.model.load_state_dict(ckpt["model_state"])
        ppo_trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        ppo_trainer.scheduler.load_state_dict(ckpt["scheduler_state"])
        ppo_trainer.entropy_coef = ckpt["entropy_coef"]
        start_update = ckpt["update_idx"] + 1
        logging.info(f"Resuming PPO from update {start_update}")
    else:
        start_update = 0
        logging.info("No PPO checkpoint found; starting from scratch")

    # 2) Run or resume training on rank 0
    if local_rank == 0:
        ppo_trainer.train(
            total_timesteps=1_000_000 // world_size,
            start_update=start_update
        )
    dist.barrier(device_ids=[local_rank])

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

    # ──────────────────────────────────────────────────────────────────────────
    # Example: processing one live bar (uncomment & adapt in production)
    # new_bar = {
    #     "date_time": pd.Timestamp.utcnow(),
    #     "Open": 3357.0, "High": 3360.0,
    #     "Low": 3355.0, "Close": 3358.0,
    #     "Volume": 120
    # }
    # df_live = process_live_row(new_bar)
    # logging.info("Live features:\n%s", df_live.head(1))

if __name__ == "__main__":
    main()
