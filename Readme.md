# Hybrid Trading Strategy: Genetic Algorithms and PPO

## Overview

This repository demonstrates a **hybrid trading strategy** combining **Genetic Algorithms (GA)** and **Proximal Policy Optimization (PPO)**, leveraging GPU acceleration for data preprocessing and training. It utilizes minute-level OHLCV market data from multiple assets (e.g., AAPL, TSLA, QQQ, SPY, futures, oil) to optimize trading strategies that maximize the **Compound Annual Growth Rate (CAGR)** and **Sharpe ratio** while minimizing **drawdowns**.

## Core Concepts

### Reinforcement Learning (RL) for Trading

An RL agent interacts with market data, taking actions such as going **long**, **short**, or **holding**. The agent is rewarded based on profit or loss, enabling it to learn an optimal trading policy.

### Genetic Algorithms (GA)

GA evolves a population of neural network policies through selection, crossover, and mutation, evaluating fitness based on historical trading performance. The top policies evolve toward more profitable and stable trading strategies.

### Policy Gradient Methods (PPO)

Proximal Policy Optimization (PPO) updates policy parameters by maximizing a clipped objective for stable and efficient training:

```
L^{CLIP}(θ) = Eₜ[min(rₜ(θ) Aₜ, clip(rₜ(θ), 1 - ε, 1 + ε) Aₜ)]
```

where:
- \( r_t(θ) = \frac{π_θ(a_t|s_t)}{π_{θ_{old}}(a_t|s_t)} \): Probability ratio
- \( A_t \): Advantage estimator

## Directory Structure

```
.
├── data_txt/
│   ├── 2000_01_SPY.txt
│   ├── 2000_01_TSLA.txt
│   └── ... (1-min OHLCV data)
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

### Step 2: Set up Conda Environment

```bash
conda create -n GeneticTrading python=3.10
conda activate GeneticTrading
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install RAPIDS GPU Acceleration

Ensure CUDA Toolkit (12.2+) is installed, then run:

```bash
conda install -c rapidsai -c nvidia -c conda-forge rapids=23.12 cudatoolkit=12.2
```

### Step 5: Data Preparation

Place minute-level OHLCV `.txt` files in the `data_txt/` folder. The required data format is:

```
date_time,open,high,low,close,volume
```

## Running the Application

Execute the main training and evaluation process with GPU acceleration:

```bash
torchrun --nproc_per_node=4 main.py
```

### Required Arguments and Environment Variables

The app leverages distributed training with PyTorch. Ensure the following environment variables are set:
- `LOCAL_RANK`: Automatically handled by `torchrun`.

### GPU Acceleration

The application uses **cuDF (RAPIDS)** for fast, GPU-accelerated data loading and preprocessing.

## Performance Metrics

- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted performance
- **Max Drawdown (MaxDD)**: Maximum observed loss from peak to trough

## Interpreting PPO Results

| Mean Reward per Step | Interpretation                        |
|----------------------|---------------------------------------|
| < 0                  | 🚩 Poor (losing strategy)              |
| 0 - 0.0001           | ⚠️ Weak profitability (near breakeven) |
| 0.0001 - 0.001       | ✅ Good, stable profitability          |
| > 0.001              | 🚀 Excellent performance               |

Example PPO output:
```
Update 0, mean reward = 0.002
...
Update 20, mean reward = 0.004
```

## Potential Extensions

- Transaction cost modeling
- Advanced position sizing
- Implementation of stop-loss and take-profit
- Additional neuroevolution methods (e.g., NEAT, CMA-ES)
- Multi-agent or multi-asset strategies

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.

