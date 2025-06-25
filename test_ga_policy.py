#!/usr/bin/env python
"""
test_ga_policy.py

Evaluate a saved GA-evolved policy on your FuturesEnv test set.
"""
import argparse
import os
import numpy as np
import torch

from ga_policy_evolution import _unpack_step as unpack_step, _unpack_reset as unpack_reset
from ga_policy_evolution import PolicyNetwork
from data_preprocessing import create_environment_data
from futures_env import FuturesEnv, TimeSeriesState

def build_states_for_futures_env(df_chunk):
    """
    Convert a pandas/cudf DataFrame slice into a list of TimeSeriesState.
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
        # append all one-hots
        for col in df_chunk.columns:
            if col.startswith("tb_") or col.startswith("wd_"):
                feats.append(getattr(row, col))
        states.append(
            TimeSeriesState(
                ts=row.date_time,
                open_price=getattr(row, "Open_raw", row.Open),
                close_price=getattr(row, "Close_raw", row.Close),
                features=feats,
            )
        )
    return states

from utils import make_test_env

def main():
    parser = argparse.ArgumentParser(description="Test a GA-trained policy")
    parser.add_argument("--model-path",   type=str, default="ga_policy_model.pth")
    parser.add_argument("--data-folder",  type=str, default="./data_txt")
    parser.add_argument("--cache-folder", type=str, default="./cached_data")
    parser.add_argument("--episodes",     type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.cache_folder, exist_ok=True)
    env = make_test_env(args.data_folder, args.cache_folder)

    input_dim  = int(np.prod(env.observation_space.shape))
    output_dim = env.action_space.n

    # load your GA policy
    policy = PolicyNetwork(input_dim, hidden_dim=64, output_dim=output_dim, device="cpu")
    policy.load_model(args.model_path)

    rewards = []
    for ep in range(args.episodes):
        reset_ret = env.reset()
        obs = unpack_reset(reset_ret)
        done = False
        total = 0.0
        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            a = policy.act(state_t)
            obs, r, done, info = unpack_step(env.step(a))
            total += r
        rewards.append(total)
        print(f"Episode {ep+1:2d} reward: {total:.2f}")

    print("=" * 40)
    print(f"Across {args.episodes} episodes →  mean: {np.mean(rewards):.2f},  std: {np.std(rewards):.2f}")

if __name__ == "__main__":
    main()
