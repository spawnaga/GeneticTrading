# utils.py

import numpy as np
import torch

def compute_performance_metrics(profit_history, timestamps, starting_capital: float = 1.0):
    """
    Compute CAGR, Sharpe ratio, and maximum drawdown from profit history.

    Args:
        profit_history: list or array of P&L values at each step.
        timestamps: list of datetime timestamps corresponding to each P&L.
        starting_capital: initial account value before applying profits.

    Returns:
        cagr: Compound Annual Growth Rate
        sharpe: annualized Sharpe ratio
        mdd: Maximum drawdown
    """
    import numpy as np

    # Need at least one profit and two timestamps to compute anything
    if len(profit_history) < 1 or len(timestamps) < 2:
        return 0.0, 0.0, 0.0

    # Build a balance series from profits
    profits = np.array(profit_history, dtype=np.float64)
    balance = starting_capital + np.cumsum(profits)

    # Avoid division by zero or negative starting capital
    if balance[0] <= 0:
        return 0.0, 0.0, 0.0

    # CAGR
    total_growth = balance[-1] / balance[0]
    elapsed_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    years = elapsed_seconds / (365.25 * 24 * 3600)
    if years > 0 and total_growth > 0:
        cagr = total_growth ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    # Period returns for Sharpe
    period_rets = np.diff(balance) / balance[:-1]
    if len(period_rets) < 1:
        return cagr, 0.0, 0.0

    # Annualization factor (skip if freq invalid)
    freq_seconds = (timestamps[1] - timestamps[0]).total_seconds()
    if freq_seconds > 0:
        ann_fac = (365.25 * 24 * 3600) / freq_seconds
        ann_mean = np.mean(period_rets) * ann_fac
        ann_std = np.std(period_rets) * np.sqrt(ann_fac)
        sharpe = ann_mean / (ann_std + 1e-8)
    else:
        sharpe = 0.0

    # Maximum drawdown
    peak = np.maximum.accumulate(balance)
    drawdowns = (peak - balance) / (peak + 1e-8)
    mdd = float(np.max(drawdowns))

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
