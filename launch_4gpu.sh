
#!/bin/bash

# 4 GPU Training Launcher
# =======================

echo "🚀 Starting 4 GPU Training with 1000 rows..."
echo "📊 Data: Limited to 1000 rows"
echo "🔥 GPUs: 4 GPUs with NVLink optimization"
echo "⚡ Mode: Adaptive GA + PPO training"
echo ""

# Check GPU availability
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "⚠️  nvidia-smi not available - running in CPU mode"

# Set permissions
chmod +x run_4gpu_1000rows.py

# Run the training
python run_4gpu_1000rows.py

echo ""
echo "✅ Training script completed!"
echo "📁 Check ./models/4gpu_1000rows/ for saved models"
echo "📈 Check ./logs/4gpu_1000rows/ for training logs"
echo "📊 Check ./runs/4gpu_1000rows/ for TensorBoard logs"
