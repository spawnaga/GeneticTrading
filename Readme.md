# 🧬 Genetic Trading System with Adaptive AI

A sophisticated algorithmic trading system that combines **Genetic Algorithms (GA)** and **Proximal Policy Optimization (PPO)** with intelligent adaptive switching based on market conditions. The system automatically selects the optimal training method based on performance patterns and market regime detection.

## 🚀 Key Features

### 🧠 Adaptive Training System
- **Intelligent Method Switching**: Automatically alternates between GA and PPO based on:
  - Performance stagnation detection
  - Market volatility analysis
  - Policy diversity monitoring
  - Regime change detection
- **Market Condition Detector**: Analyzes volatility, trends, volume, and microstructure
- **Performance Tracking**: Continuous monitoring with automatic adaptation

### 🔬 Advanced Optimization Methods
- **Genetic Algorithm (GA)**: Population-based evolution for exploration
- **Proximal Policy Optimization (PPO)**: Gradient-based refinement for exploitation
- **Distributed Training**: Multi-GPU support with automatic load balancing
- **GPU Acceleration**: RAPIDS cuDF integration for high-performance data processing

### 📊 Professional Trading Environment
- **Futures Trading Simulation**: Realistic tick-by-tick execution
- **Advanced Market Features**: Bid-ask spreads, slippage, commission modeling
- **Risk Management**: Margin requirements and position sizing
- **Performance Metrics**: CAGR, Sharpe Ratio, Maximum Drawdown

## 🎯 Training Modes

### 1. Adaptive Mode (Recommended)
```bash
python main.py --training-mode adaptive --adaptive-iterations 20
```
Intelligently switches between GA and PPO based on market conditions and performance.

### 2. GA Only Mode
```bash
python main.py --training-mode ga_only --ga-generations 100 --ga-population 80
```
Pure evolutionary approach for exploration-heavy scenarios.

### 3. PPO Only Mode
```bash
python main.py --training-mode ppo_only --total-steps 1000000
```
Gradient-based optimization for fine-tuning existing policies.

### 4. Sequential Mode
```bash
python main.py --training-mode sequential
```
Traditional approach: GA first, then PPO refinement.

## 📁 Project Structure

```
GeneticTrading/
├── 🗂️ Core System
│   ├── main.py                    # Main orchestrator with adaptive training
│   ├── adaptive_trainer.py        # Intelligent GA/PPO switching logic
│   ├── market_condition_detector.py # Market regime analysis
│   └── model_manager.py          # Advanced model management
├── 🧬 Algorithms
│   ├── ga_policy_evolution.py     # Genetic Algorithm implementation
│   ├── policy_gradient_methods.py # PPO implementation
│   └── futures_env.py            # Professional trading environment
├── 📊 Data & Preprocessing
│   ├── data_preprocessing.py      # GPU-accelerated data pipeline
│   ├── data_txt/                 # Raw market data files
│   └── cached_data/              # Processed data cache
├── 💾 Models & Logs
│   ├── models/                   # Trained models with backups
│   │   ├── ga_models/
│   │   ├── ppo_models/
│   │   ├── checkpoints/
│   │   └── backups/
│   └── logs/                     # Training and evaluation logs
└── 📈 Analysis
    └── runs/                     # TensorBoard logs
```

## 🔧 Installation & Setup

### For Replit (Recommended)
Simply click the **Run** button! All dependencies are automatically managed.

### Key Dependencies
- **PyTorch**: Deep learning framework
- **NumPy/Pandas**: Data manipulation
- **Gymnasium**: RL environment interface
- **Matplotlib**: Visualization
- **TensorBoard**: Training monitoring
- **RAPIDS cuDF**: GPU-accelerated data processing (optional)

## ⚙️ Configuration Options

### Data Configuration
```bash
--data-folder ./data_txt           # Path to OHLCV data files
--cache-folder ./cached_data       # Processed data cache
--max-rows 50000                   # Maximum rows to process
--data-percentage 1.0              # Percentage of data to use
```

### Training Configuration
```bash
--adaptive-iterations 20           # Adaptive training iterations
--stagnation-threshold 5           # Switch to GA after N stagnant iterations
--poor-performance-threshold 3     # Switch to GA after N poor performances
--ga-population 80                 # GA population size
--ga-generations 100               # GA generations
--ppo-lr 3e-4                     # PPO learning rate
```

### Environment Configuration
```bash
--value-per-tick 12.5             # Contract value per tick
--tick-size 0.25                  # Minimum price increment
--commission 0.0005               # Commission per trade
--margin-rate 0.01                # Margin requirement
```

## 📊 Understanding the Adaptive System

### When GA is Preferred:
- **High Volatility Markets**: Exploration needed for changing conditions
- **Performance Stagnation**: Current policy not improving
- **Low Policy Diversity**: Policy becoming too deterministic
- **Market Regime Changes**: Major shifts in market behavior

### When PPO is Preferred:
- **Stable Markets**: Fine-tuning for consistent conditions
- **Good Performance**: Refining profitable strategies
- **Post-GA Optimization**: Gradient-based improvement after GA exploration

### Market Condition Detection:
- **Volatility Analysis**: Current vs historical volatility ratios
- **Trend Detection**: Linear regression with strength measurement
- **Volume Patterns**: Increasing/decreasing volume trends
- **Microstructure**: Momentum vs mean-reversion detection

## 📈 Performance Metrics

### Financial Metrics
- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

### Training Metrics
- **Policy Entropy**: Measure of exploration vs exploitation
- **Performance Stability**: Variance in recent performance
- **Convergence Speed**: Rate of improvement over time

## 🚀 Quick Start Examples

### Basic Adaptive Training
```bash
python main.py
```

### High-Performance Training
```bash
python main.py --max-rows 100000 --ga-population 100 --adaptive-iterations 30
```

### Market-Specific Training
```bash
python main.py --data-folder ./high_vol_data --stagnation-threshold 3
```

## 🔍 Monitoring & Analysis

### Real-Time Monitoring
- **Console Logs**: Detailed training progress
- **TensorBoard**: Visual training metrics
- **Performance Tracking**: Automatic metric computation

### Model Management
- **Automatic Backups**: Timestamped model saves
- **Checkpoint Recovery**: Resume from interruptions
- **Best Model Selection**: Automatic performance-based selection

## 🛠️ Advanced Features

### Distributed Training
- **Multi-GPU Support**: Automatic GPU detection and utilization
- **Process Synchronization**: NCCL backend for efficient communication
- **Load Balancing**: Optimal data distribution across GPUs

### Data Pipeline
- **GPU Acceleration**: RAPIDS cuDF for high-performance processing
- **Intelligent Caching**: Avoid redundant preprocessing
- **Memory Management**: Chunked processing for large datasets

## 📚 Key Algorithms Explained

### Genetic Algorithm (GA)
- **Population Evolution**: Maintains diverse policy population
- **Selection Pressure**: Elite preservation with tournament selection
- **Crossover & Mutation**: Parameter space exploration
- **Fitness Evaluation**: Comprehensive trading performance assessment

### Proximal Policy Optimization (PPO)
- **Actor-Critic Architecture**: Policy and value function optimization
- **Clipped Objectives**: Stable policy updates
- **Generalized Advantage Estimation**: Improved gradient estimates
- **Mini-batch Training**: Efficient GPU utilization

### Adaptive Logic
- **Performance Monitoring**: Continuous metric tracking
- **Regime Detection**: Market condition analysis
- **Switch Decisions**: Rule-based method selection
- **Hyperparameter Adaptation**: Dynamic parameter adjustment

## 🚧 Troubleshooting

### Common Issues
- **GPU Memory**: Reduce batch sizes or population size
- **Slow Training**: Check data preprocessing cache
- **Poor Performance**: Increase training data or adjust parameters

### Performance Optimization
- **Data Size**: Use `--max-rows` to limit dataset size for testing
- **Cache Utilization**: Ensure cached data is being used
- **GPU Usage**: Monitor with `nvidia-smi` (if available)

## 🎯 Best Practices

1. **Start Small**: Test with limited data before full training
2. **Monitor Logs**: Watch for performance patterns and switches
3. **Adjust Thresholds**: Tune switching parameters for your data
4. **Save Regularly**: Use automatic backup features
5. **Evaluate Thoroughly**: Test on out-of-sample data

## 📝 Recent Updates

- ✅ **Adaptive Training System**: Intelligent GA/PPO switching
- ✅ **Market Condition Detection**: Automatic regime analysis
- ✅ **Enhanced Model Management**: Backup and recovery systems
- ✅ **Professional Configuration**: Comprehensive argument parsing
- ✅ **Performance Optimization**: Memory-efficient data processing
- ✅ **Distributed Training**: Multi-GPU support improvements

---

## 🎉 Getting Started

Ready to start? Simply click the **Run** button in Replit, or use:

```bash
python main.py --training-mode adaptive
```

The system will automatically detect your hardware capabilities, process your data, and begin adaptive training with intelligent switching between genetic algorithms and reinforcement learning based on market conditions and performance patterns.

Happy Trading! 📈🤖