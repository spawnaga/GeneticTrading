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

   ```
   L^{CLIP}(θ) = Eₜ[min(rₜ(θ) Aₜ, clip(rₜ(θ), 1 - ε, 1 + ε) Aₜ)]
   ```

   where \(r_t(θ) = \frac{π_θ(a_t|s_t)}{π_{θ_old}(a_t|s_t)}\) is the probability ratio between the new and old policies, and \(A_t\) is an advantage estimator (e.g., GAE-lambda).

3. **Performance Metrics**  
   - **CAGR** (Compound Annual Growth Rate): annualized growth rate of equity.
   - **Sharpe Ratio** measures risk-adjusted returns.
   - **Max Drawdown (MDD)** quantifies the worst peak-to-trough decline.

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

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourname/hybrid-trading-rl.git
   cd hybrid-trading-rl
   ```

2. **Create a Virtual Environment (Recommended)**
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
   - Columns format: `date_time,open,high,low,close,volume`.
   - Data should span from 2000 to 2024.

## Usage

1. **Data Preprocessing**  
   The data is automatically loaded and transformed when you run `main.py`. Specifically, `create_environment_data('./data_txt')` in **`data_preprocessing.py`**:
   - Concatenates all `.txt` files in `data_txt/`.
   - Sorts by `date_time`.
   - Computes simple return, moving averages, etc.
   - Scales features using `StandardScaler`.
   - Splits into an 80/20 train/test partition.

2. **Run GA + PPO Training**  
   Execute:
   ```bash
   python main.py
   ```
   This runs GA evolution, PPO training, evaluates agents on test data, computes performance metrics, and visualizes equity curves.

3. **Multi-GPU Setup**  
   Single GPU by default. Extendable to multi-GPU with DataParallel or DistributedDataParallel.

## Interpreting PPO Performance

| Mean Reward per Step | Interpretation                        |
|----------------------|---------------------------------------|
| < 0                  | 🚩 Poor (losing strategy)              |
| 0 - 0.0001           | ⚠️ Weak profitability (near breakeven) |
| 0.0001 - 0.001       | ✅ Good, stable profitability          |
| > 0.001              | 🚀 Excellent performance               |

Good PPO training results might look like:
```
Update 0, mean reward = 0.002
...
Update 20, mean reward = 0.004
```

3. **Multi-GPU Setup**  
   Single GPU by default. Extendable to multi-GPU with DataParallel or DistributedDataParallel from PyTorch. Or parallelize GA evaluations across GPUs. Advanced configuration required.

## Key Math Details

### GA Fitness Function
Fitness = cumulative reward:
```
fitness(θ) = Σₜ rₜ(θ)
```

### PPO Objective
```
L^{CLIP}(θ) = Eₜ[min(rₜ(θ) Aₜ, clip(rₜ(θ), 1-ε, 1+ε) Aₜ)]
```

### CAGR, Sharpe, Max Drawdown
- **CAGR**: `(Final Equity / Initial Equity)^(1/T) - 1`
- **Sharpe**: `(Mean Return - Risk-Free Rate) / Std Dev of Returns`
- **Max Drawdown**: `max(peak - current_balance) / Peak`

## Rendering Math Equations on GitHub
GitHub Markdown doesn't render LaTeX directly. Recommended solutions:
- Use GitHub Pages with Jekyll and MathJax for rendered equations.
- Convert equations to images.
- Clearly represent equations in markdown code blocks.

## Potential Extensions
- **Transaction Costs**
- **Advanced Position Sizing**
- **Stop Loss/Take Profit**
- **Neuroevolution (e.g., NEAT, CMA-ES)**
- **Multi-Agent/Multi-Asset Trading**

## Contributing
Feel free to open issues or submit pull requests for improvements:
- Enhanced data ingestion methods.
- Parallel training setups.
- Improved logging (e.g., TensorBoard, Weights & Biases).

## License
Distributed under the **MIT License**. Free for use, modification, and distribution with proper attribution.

---