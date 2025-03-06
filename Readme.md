# Hybrid Trading Strategies Using Genetic Algorithms and PPO

**Author**: Alex Oraibi  
**License**: MIT License  

---

## Overview

This repository showcases a **hybrid trading strategy** combining **Genetic Algorithms (GA)** and **Proximal Policy Optimization (PPO)** to develop profitable trading strategies. The model is trained using minute-level OHLCV (Open, High, Low, Close, Volume) data across multiple financial instruments, including stocks (e.g., AAPL, TSLA, QQQ, SPY), futures, and commodities (e.g., oil).

### Goals:
- Maximize **Compound Annual Growth Rate (CAGR)**
- Maximize **Sharpe Ratio**
- Minimize **Maximum Drawdown (MDD)**

---

## Key Concepts

### 1. Reinforcement Learning (RL) in Trading
The RL agent learns by interacting with a market environment, observing market features such as price trends and technical indicators (e.g., moving averages, RSI). Actions taken by the agent are:
- **Hold**
- **Long (Buy)**
- **Short (Sell)**

The agent’s goal is to optimize a policy that maximizes cumulative rewards, which typically represent profits, while managing risk.

### 2. Genetic Algorithms (GA)
GA evolves a population of candidate neural network policies through iterative cycles involving:
- **Evaluation:** Assessing performance (fitness) based on cumulative reward.
- **Selection:** Best-performing policies reproduce.
- **Crossover and Mutation:** Generate new policy variants.

### GA Fitness Function
\[ \text{Fitness}(\theta) = \sum_{t} r_t(\theta) \]

### GA Process
- Evaluate fitness on historical data.
- Select top performers.
- Apply genetic operations to evolve the population.

### Proximal Policy Optimization (PPO)
PPO refines the evolved policies using a policy gradient approach optimized via a clipped objective:

\[ L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right) \right] \]

where:
- \( r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \) (probability ratio)
- \( A_t \) is advantage estimation (using GAE)
- \( \epsilon \) controls clipping (typically 0.2)

## Performance Metrics
- **Compound Annual Growth Rate (CAGR):**
\[ \text{CAGR} = \left(\frac{\text{Final Equity}}{\text{Initial Equity}}\right)^{\frac{1}{T}} - 1 \]

- **Sharpe Ratio:**
\[ \text{Sharpe Ratio} = \frac{\text{Mean Return} - \text{Risk-Free Rate}}{\text{Std. Dev. of Returns}} \]

- **Max Drawdown (MDD):**
\[ \text{MDD} = \max \left( \frac{\text{Peak Equity} - \text{Current Equity}}{\text{Peak Equity}} \right) \]

## Directory Structure
```
.
├── data_txt/
│   ├── 2000_01_SPY.txt
│   ├── 2000_01_TSLA.txt
│   └── ...
├── data_preprocessing.py
├── trading_environment.py
├── ga_policy_evolution.py
├── policy_gradient_methods.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AlexOraibi/hybrid-trading-strategy.git
cd hybrid-trading-rl
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
Place minute-level OHLCV `.txt` or `.csv` files into the `data_txt/` directory.

Format required:
```
date_time,open,high,low,close,volume
```

---

## Usage

### Run the training process:
```bash
python main.py
```

### Multi-GPU Support
To leverage multiple GPUs:
```bash
torchrun --nproc_per_node=4 main.py
```

---

## PPO Training Interpretation

| Mean Reward per Step | Performance                        |
|----------------------|------------------------------------|
| < 0                  | 🚩 Poor                             |
| 0 - 0.0001           | ⚠️ Marginal                         |
| 0.0001 - 0.001       | ✅ Stable Profitability             |
| > 0.001              | 🚀 Excellent                        |

Example training output:
```
Update 0, mean reward = 0.002
...
Update 20, mean reward = 0.004
```

---

## Directory Details
- **data_preprocessing.py:** Loads, preprocesses, and splits data.
- **trading_environment.py:** Defines the RL environment.
- **ga_policy_evolution.py:** Handles genetic algorithm processes.
- **policy_gradient_methods.py:** Implements PPO training algorithm.
- **main.py:** Entry point for training and evaluation.

---

## Extensions & Future Development
- Incorporate **transaction costs** for realistic backtesting.
- Implement **dynamic position sizing** and risk management.
- Integrate automated **stop-loss** and **take-profit** mechanisms.
- Explore alternative evolutionary methods like **NEAT** or **CMA-ES**.
- Expand to **multi-agent**, **multi-asset** strategies.

---

## Rendering Mathematical Equations
- Use GitHub Pages with **MathJax** for real-time rendering.
  - Add to your `_config.yml`:
```yaml
markdown: kramdown
kramdown:
  math_engine: mathjax
```

Alternatively, equations can be displayed as plain text in markdown code blocks or converted into images for static embedding.

---

## Contributing
Contributions are welcome! Please follow these guidelines:
- Open an issue to discuss new features or bug fixes.
- Submit pull requests clearly describing improvements.
- Enhance data ingestion, parallel training, or add advanced logging (TensorBoard, Weights & Biases).

---

## License
This project is distributed under the **MIT License**—free to use, modify, and redistribute with proper attribution.

---

### Author
**Alex Oraibi**  
[GitHub Profile](https://github.com/AlexOraibi)

© 2025 Alex Oraibi - MIT License