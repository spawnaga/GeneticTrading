# utils.py

import numpy as np
import torch

def compute_performance_metrics(balance_history, timestamps):
    """
    Compute CAGR, Sharpe ratio, and maximum drawdown from profit history.
    """
    if len(balance_history) < 2:
        return 0.0, 0.0, 0.0

    bh     = np.array(balance_history, dtype=np.float64)
    total  = bh[-1] / (bh[0] + 1e-8)
    elapsed= (timestamps[-1] - timestamps[0]).total_seconds()
    years  = elapsed / (365.25 * 24 * 3600)
    cagr   = total**(1/years) - 1 if years > 0 else 0.0

    rets   = np.diff(bh) / (bh[:-1] + 1e-8)
    freq   = (timestamps[1] - timestamps[0]).total_seconds()
    ann_fac= (365.25 * 24 * 3600) / freq
    ann_mean = np.mean(rets) * ann_fac
    ann_std  = np.std(rets) * np.sqrt(ann_fac)
    sharpe   = ann_mean / (ann_std + 1e-8)

    peak      = np.maximum.accumulate(bh)
    drawdowns= (peak - bh) / (peak + 1e-8)
    mdd       = np.max(drawdowns)

    return cagr, sharpe, mdd


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
