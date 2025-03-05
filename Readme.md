# Hybrid Trading Strategy: Genetic Algorithms and PPO

## Overview

This repository demonstrates a **hybrid approach** combining **Genetic Algorithms (GA)** and **Policy Gradient** methods (specifically **PPO**) to evolve trading strategies using GPU-accelerated data processing. It operates on minute-level data spanning multiple assets (e.g., AAPL, TSLA, QQQ, SPY, futures, oil), aiming to maximize **CAGR (Compound Annual Growth Rate)** and **Sharpe ratio** while minimizing **drawdowns**.

## Core Concepts

### Reinforcement Learning (RL) for Trading
An agent interacts with the market by observing market data and acting (long, short, hold). Rewards represent profit or loss. The agent learns an optimal policy to maximize returns while managing risk.

### Genetic Algorithms (GA)
GA evolves a population of candidate solutions (neural network policies). Policies are evaluated on historical data to compute fitness. Top performers are selected to produce offspring via crossover and mutation, evolving towards optimal trading policies.

### Policy Gradient (PPO)
Proximal Policy Optimization (PPO) updates policy parameters by maximizing a clipped objective to ensure stable training:

```
L^{CLIP}(θ) = Eₜ[min(rₜ(θ) Aₜ, clip(rₜ(θ), 1 - ε, 1 + ε) Aₜ)]
```

where:
- \( r_t(θ) = \frac{π_θ(a_t|s_t)}{π_{θ_{old}}(a_t|s_t)} \): Probability ratio between new and old policies
- \( A_t \): Advantage estimator

## Directory Structure

```
.
├── data_txt/
│   ├── 2000_01_SPY.txt
│   ├── 2000_01_TSLA.txt
│   └── ... (1-min OHLCV data in CSV format)
├── cached_data/ (GPU-generated cached Parquet files)
├── data_preprocessing.py
├── trading_environment.py
├── ga_policy_evolution.py
├── policy_gradient_methods.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourname/hybrid-trading-rl.git
cd hybrid-trading-rl
```

### Step 2: Set up a Conda Environment

Use conda to create and activate the environment:

```bash
conda create -n GeneticTrading python=3.10
conda activate GeneticTrading
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: RAPIDS GPU Acceleration Setup

Ensure NVIDIA CUDA Toolkit (12.2+) is installed, then install RAPIDS libraries:

```bash
conda install -c rapidsai -c nvidia -c conda-forge rapids=23.12 cudatoolkit=12.2
```

### Step 5: Data Preparation

- Place your minute-level `.txt` OHLCV files in `data_txt/`.
- Data format: `date_time,open,high,low,close,volume`

## Running the Application

To preprocess data, evolve strategies, and train with PPO:

```bash
python main.py
```

### GPU Acceleration

The application uses GPU acceleration (cuDF) for efficient data preprocessing and feature engineering.

## Performance Metrics

- **CAGR**: Annualized growth rate.
- **Sharpe Ratio**: Risk-adjusted returns.
- **Max Drawdown**: Worst peak-to-trough decline.

## Interpreting PPO Results

| Mean Reward per Step | Interpretation                        |
|----------------------|---------------------------------------|
| < 0                  | 🚩 Poor (losing strategy)              |
| 0 - 0.0001           | ⚠️ Weak profitability (near breakeven) |
| 0.0001 - 0.001       | ✅ Good, stable profitability          |
| > 0.001              | 🚀 Excellent performance               |

Example PPO training output:
```
Update 0, mean reward = 0.002
...
Update 20, mean reward = 0.004
```

## Potential Extensions

- Transaction cost modeling
- Advanced position sizing
- Stop-loss and take-profit implementation
- Neuroevolution methods (e.g., NEAT, CMA-ES)
- Multi-agent or multi-asset strategies

## Contributing

Contributions and improvements are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.

