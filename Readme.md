# Professional Trading System with GA and PPO

A sophisticated algorithmic trading system that combines Genetic Algorithm (GA) policy evolution with Proximal Policy Optimization (PPO) for adaptive strategy learning.

## 🚀 Quick Start

### For PyCharm Users

1. **Clone and Open Project**
   ```bash
   git clone <your-repo-url>
   cd GeneticTrading
   ```

2. **Install Dependencies**
   ```bash
   pip install torch numpy pandas scikit-learn matplotlib tensorboard
   pip install gym stable-baselines3
   ```

3. **Run Simple Test**
   ```bash
   python run_simple.py
   ```
   Or in PyCharm: Right-click `run_simple.py` → Run

4. **Run Development Mode**
   ```bash
   python run_simple.py dev
   ```

### For Advanced Users

#### Single GPU/CPU Training
```bash
python main.py --data-percentage 0.1 --max-rows 5000
```

#### Distributed Training (4 GPUs)
```bash
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12355 main.py --data-percentage 1.0 --models-dir ./models/4gpu_production --total-steps 5000000
```

## 📁 Project Structure

```
GeneticTrading/
├── main.py                 # Main training script
├── run_simple.py           # Simple launcher for PyCharm
├── adaptive_trainer.py     # Adaptive GA+PPO trainer
├── futures_env.py          # Trading environment
├── ga_policy_evolution.py  # Genetic algorithm implementation
├── policy_gradient_methods.py # PPO implementation
├── data_preprocessing.py   # Data processing pipeline
├── utils.py               # Utility functions
├── models/                # Saved models
├── logs/                  # Training logs
├── cached_data/           # Processed data cache
└── data_txt/             # Raw market data
```

## 🔧 Configuration

### Quick Configuration Options

**Test Mode (Fast):**
- Data: 1% of dataset (~500 rows)
- Training: 1,000 steps
- GA Population: 10
- Perfect for debugging and quick tests

**Development Mode:**
- Data: 10% of dataset (~5,000 rows) 
- Training: 50,000 steps
- GA Population: 20
- Good for feature development

**Production Mode:**
- Data: 100% of dataset
- Training: 1,000,000+ steps
- GA Population: 80+
- Full-scale training

### Advanced Configuration

Edit command line arguments in `main.py` or use configuration files:

```bash
python main.py \
  --data-percentage 0.5 \
  --max-rows 50000 \
  --models-dir ./models/custom \
  --total-steps 500000 \
  --ga-population 50 \
  --ppo-lr 0.0003
```

## 🎯 Key Features

- **Adaptive Training**: Intelligent switching between GA and PPO based on performance
- **Distributed Training**: Multi-GPU support with automatic load balancing  
- **Professional Logging**: Comprehensive logging with TensorBoard integration
- **Robust Error Handling**: Graceful handling of CUDA, data, and training errors
- **Modular Design**: Easy to extend and customize components

## 📊 Monitoring

### TensorBoard
```bash
tensorboard --logdir=./runs
```

### Log Files
- Training logs: `./logs/`
- Model checkpoints: `./models/`
- Performance metrics: Console output

## 🛠 Development Workflow

1. **Start with Test Mode**
   ```bash
   python run_simple.py test
   ```

2. **Develop Features in Dev Mode**
   ```bash
   python run_simple.py dev
   ```

3. **Run Full Training**
   ```bash
   python main.py --data-percentage 1.0
   ```

4. **Monitor with TensorBoard**
   ```bash
   tensorboard --logdir=./runs
   ```

## 📋 Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 1.12+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- TensorBoard

### Optional (for GPU acceleration)
- CUDA 11.6+
- cuDF (RAPIDS)
- cuML (RAPIDS)

### Development Tools
- PyCharm (recommended)
- Jupyter Notebook
- Git

## 🚨 Troubleshooting

### Common Issues

**CUDA/GPU Issues:**
- System falls back to CPU automatically
- Check CUDA installation: `nvidia-smi`

**Memory Issues:**
- Reduce `--data-percentage` 
- Reduce `--max-rows`
- Reduce `--ga-population`

**Import Errors:**
- Install missing packages: `pip install <package>`
- Check Python version compatibility

**Training Crashes:**
- Check log files in `./logs/`
- Reduce data size for debugging
- Use `run_simple.py test` for minimal testing

## 📈 Performance Tips

1. **For Fast Iteration**: Use test mode (`run_simple.py`)
2. **For GPU Training**: Use 4-GPU distributed mode
3. **For Memory Efficiency**: Adjust batch sizes and data percentage
4. **For Stability**: Monitor logs and use checkpointing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Test with: `python run_simple.py test`
4. Commit changes: `git commit -am 'Add feature'`
5. Push branch: `git push origin feature-name`
6. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.