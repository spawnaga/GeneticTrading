

# README

## Overview

This repository demonstrates a **hybrid approach** that applies both **Genetic Algorithms (GA)** and **Policy Gradient** methods (specifically **PPO**) to **evolve trading strategies** on minute-level data spanning multiple assets (e.g., APPL, TSLA, QQQ, SPY, futures, oil). We aim to maximize **CAGR (Compound Annual Growth Rate)** and **Sharpe ratio** while minimizing **drawdowns**.

### Core Concepts

1. **Reinforcement Learning (RL) for Trading**  
   In RL-based trading, an agent interacts with a market environment by **observing** market features (open, high, low, close, volume, technical indicators) and **acting** (long, short, hold). The **reward** typically represents profit or loss (PnL). Over time, the agent learns an optimal policy to maximize cumulative returns while controlling risk.

2. **Genetic Algorithms (GA)**  
   A GA evolves a **population** of candidate solutions (here, neural network policies). Each policy is evaluated on historical data to get a **fitness** (total reward). Then, top performers are **selected** as parents to produce offspring via **crossover** and **mutation**. Over generations, the population converges to high-return policies.

3. **Policy Gradient (PPO)**  
   Proximal Policy Optimization (PPO) is a gradient-based RL method that updates the policy’s parameters \(\theta\) by maximizing a clipped objective that prevents overly large policy updates. The PPO objective for each time step \(t\) can be written as:

   \[
   L^{CLIP}(\theta) \;=\; \mathbb{E}_t\biggl[
     \min\Bigl( r_t(\theta) \, A_t,\;
       \operatorname{clip}\bigl(r_t(\theta), 1-\epsilon, 1+\epsilon\bigr) \, A_t
     \Bigr)
   \biggr],
   \]

   where \(r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}\) is the probability ratio between the new and old policies, and \(A_t\) is an advantage estimator (e.g., GAE-lambda).

4. **Performance Metrics**  
   - **CAGR** (Compound Annual Growth Rate) estimates the annualized growth of the equity curve:
     \[
     \text{CAGR} = \left( \frac{\text{Final Equity}}{\text{Initial Equity}} \right)^{\frac{1}{T}} - 1,
     \]
     where \(T\) is the number of years in the backtest.  
   - **Sharpe Ratio** measures risk-adjusted returns:
     \[
     \text{Sharpe} = \frac{E[R_p - R_f]}{\sigma_p},
     \]
     where \(R_p\) is the portfolio return, \(R_f\) is the risk-free rate, and \(\sigma_p\) is the standard deviation of \(R_p\).  
   - **Max Drawdown (MDD)** quantifies the worst peak-to-trough decline:
     \[
     \text{MDD} = \max_{\tau \in [0, T]} \Bigl(\frac{\text{Peak} - \text{Equity}(\tau)}{\text{Peak}}\Bigr).
     \]
     We typically want to minimize MDD, as it reflects downside risk.

## Directory Structure

```
.
├── data_txt/
│   ├── 2000_01_SPY.txt
│   ├── 2000_01_TSLA.txt
│   └── ...  (1-min OHLCV data in CSV format)
├── data_preprocessing.py
├── trading_environment.py
├── ga_policy_evolution.py
├── policy_gradient_methods.py
├── main.py
├── requirements.txt
└── README.md
```

1. **`data_txt/`** – Contains 1-minute bar data in `.txt` files. Each file is comma-separated with columns:  
   ```
   date_time,open,high,low,close,volume
   ```
   Example:
   ```
   2008-01-02 06:00:00,1527.0,1528.5,1526.75,1528.25,2317
   2008-01-02 06:01:00,1528.0,1528.5,1527.75,1528.5,777
   ...
   ```

2. **`data_preprocessing.py`** – Loads and cleans data from the `.txt` files, applies feature engineering (e.g., returns, moving averages), and splits data into **train** and **test** sets.

3. **`trading_environment.py`** – Implements a custom trading environment that:
   - Iterates through time-series data step by step.
   - Accepts actions \(\{ \text{hold}=0, \text{long}=1, \text{short}=2 \}\).
   - Calculates rewards based on the mark-to-market PnL.
   - Terminates when the end of the dataset is reached.

4. **`ga_policy_evolution.py`** – Contains:
   - A **PyTorch** `PolicyNetwork` for discrete action selection.
   - A GA loop (`run_ga_evolution`) to evolve a population of these networks:
     1. **Evaluate** each policy’s fitness by summing rewards over an episode.
     2. **Select** top elites and **crossover** + **mutate** them to form a new generation.
     3. Repeat for specified generations to find the best individual.

5. **`policy_gradient_methods.py`** – Implements a simplified **PPO** trainer (`PPOTrainer`) with:
   - **Actor-Critic** network (`ActorCriticNet`) for policy logits and state-value.
   - **Rollout** method for collecting experience.
   - **Advantages** computation (GAE-lambda).
   - **Clipped PPO objective** for stable policy updates.

6. **`main.py`** – Combines everything:
   - Loads data, creates train/test environments.
   - Runs GA evolution and PPO training.
   - Evaluates the best agents on test data.
   - Calculates **CAGR**, **Sharpe ratio**, and **Max Drawdown** for performance analysis.
   - Plots equity curves for comparison.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourname/hybrid-trading-rl.git
   cd hybrid-trading-rl
   ```

2. **Create a Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Data Setup**
   - Place your 1-minute `.txt` files (OHLCV) in the `data_txt/` directory.
   - Ensure the columns follow the format: `date_time,open,high,low,close,volume` (comma-separated).
   - Data should span from 2000 to 2024 for various tickers (e.g., AAPL, TSLA, QQQ, SPY, futures, oil).

## Usage

1. **Data Preprocessing**  
   The data is automatically loaded and transformed when you run `main.py`. Specifically, `create_environment_data('./data_txt')` in **`data_preprocessing.py`**:
   - Concatenates all `.txt` files in `data_txt/`.
   - Sorts by `date_time`.
   - Computes simple return, moving averages, etc.
   - Scales features using `StandardScaler`.
   - Splits into an 80/20 train/test partition.

2. **Run GA + PPO Training**  
   Simply run:
   ```bash
   python main.py
   ```
   This will:
   - Instantiate a **train** and **test** environment from the splitted data.
   - Run **GA** evolution for a specified number of generations (default: 10).
   - Print the best GA fitness and evaluate on test data for metrics.
   - Run **PPO** training for a certain number of timesteps (default: 20k).
   - Evaluate the trained PPO on the test set, computing performance metrics.
   - Plot the final equity curves for both GA and PPO agents.

3. **Multi-GPU Setup**  
   This demo uses a single GPU if available (`device = 'cuda'`). For multiple GPUs with NVLink, you can adapt:
   - **DataParallel** or **DistributedDataParallel** from PyTorch.
   - Or parallelize GA evaluations across GPUs, each evaluating a subset of the population.
   This requires more advanced configuration, but the codebase is structured to allow expansions.

## Key Math Details

### GA Fitness Function
The **fitness** of an individual (policy network) is the cumulative reward:
\[
\text{fitness}(\theta) = \sum_{t=0}^{T-1} r_t(\theta),
\]
where \(r_t(\theta)\) is the PnL-based reward at step \(t\).

### PPO Objective
PPO updates policy parameters \(\theta\) by optimizing the clipped objective for each step \(t\):
\[
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \Bigl[
\min\Bigl(r_t(\theta) A_t,\; \operatorname{clip}\bigl(r_t(\theta),\,1-\epsilon,\,1+\epsilon\bigr)\,A_t\Bigr)
\Bigr],
\]
where
\[
r_t(\theta) \;=\; \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}, 
\quad 
A_t \;=\; \sum_{l=0}^{\infty}\gamma^l\,\delta_{t+l},
\]
and \(\delta_{t} = r_t + \gamma V(s_{t+1}) - V(s_t)\).

### CAGR, Sharpe, Max Drawdown
1. **CAGR**:
   \[
   \text{CAGR} = \bigl(\frac{\text{Final Equity}}{\text{Initial Equity}}\bigr)^{1/T} - 1,
   \]
   with \(T\) = number of years spanned.

2. **Sharpe Ratio**:
   \[
   \text{Sharpe} = \frac{E[R_p - R_f]}{\sigma_p},
   \]
   typically annualized if the data is minute-level.

3. **Max Drawdown**:
   \[
   \text{MDD} = \max_{t} \Bigl(\frac{\text{Peak}_{t} - \text{Balance}(t)}{\text{Peak}_{t}}\Bigr),
   \]
   where \(\text{Peak}_{t} = \max_{\tau \le t}\{\text{Balance}(\tau)\}\).

## Results

After running `main.py`, you’ll see **fitness** evolution logs for GA and **training** logs for PPO. On the test set, you’ll get:

- **CAGR**: Compound annual growth > 0 is a sign of profitable strategy.
- **Sharpe**: Values above 1.0 are often considered good, above 2.0 very good, etc.
- **Max Drawdown**: Typically want to keep under 20–30% for practical risk management.

A sample console output might look like:
```
Gen 0 | Best fit: 100.50, Avg fit: 75.20, Overall best: 100.50
Gen 1 | Best fit: 110.75, Avg fit: 85.12, Overall best: 110.75
...
GA best train fitness = 180.00
GA Results - CAGR: 0.3452, Sharpe: 1.4512, MaxDD: 0.1573

Update 0, mean reward = 2.12
...
Update 9, mean reward = 2.85
PPO Results - CAGR: 0.4123, Sharpe: 1.6720, MaxDD: 0.1450
```
Finally, a **matplotlib** plot will pop up with the equity curves of GA vs PPO over the test period.

## Potential Extensions

- **Transaction Costs**: Deduct fees/spreads from reward, e.g. `reward -= cost`.
- **Position Sizing**: E.g., scaling in/out positions, or allowing partial shares.
- **Stop Loss / Take Profit**: Additional actions or constraints in the environment.
- **Neuroevolution**: Use advanced evolutionary methods like NEAT or CMA-ES.
- **Multi-Agent / Multi-Asset**: Agents that coordinate or trade multiple assets concurrently.

## Contributing

Feel free to open **issues** or submit **pull requests** for improvements:
- **Data ingestion** (multi-threaded or streaming).
- **Parallel training** with HPC or cloud-based solutions.
- **Enhanced logging** (e.g., TensorBoard, Weights & Biases).

## License

This project is distributed under the **MIT License**. You’re free to use, adapt, and share for academic or commercial purposes, provided that credit is given.

---

