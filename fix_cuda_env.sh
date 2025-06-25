
#!/bin/bash

echo "🔧 Fixing CUDA Environment Issues..."

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  Running as root, some paths may differ"
fi

# Check NVIDIA driver
echo "📋 Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    echo "Running nvidia-smi..."
    if nvidia-smi &> /dev/null; then
        nvidia-smi
        echo "✅ NVIDIA driver found and working"
    else
        echo "❌ NVIDIA driver found but not working properly"
        echo "This might be due to permission issues or driver problems"
    fi
else
    echo "❌ nvidia-smi command not found"
    echo "NVIDIA drivers may not be installed or not in PATH"
    echo "Continuing with CPU-only setup..."
fi

# Find CUDA libraries
echo "📋 Searching for CUDA libraries..."
CUDA_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/lib64"
    "/opt/cuda/lib64"
)

LIBCUDA_PATH=""
for path in "${CUDA_PATHS[@]}"; do
    if [ -f "$path/libcuda.so.1" ]; then
        LIBCUDA_PATH="$path/libcuda.so.1"
        echo "✅ Found libcuda.so.1 at: $LIBCUDA_PATH"
        break
    fi
done

if [ -z "$LIBCUDA_PATH" ]; then
    echo "❌ libcuda.so.1 not found in standard locations"
    echo "Searching system-wide..."
    LIBCUDA_PATH=$(find /usr -name "libcuda.so.1" 2>/dev/null | head -1)
    if [ -n "$LIBCUDA_PATH" ]; then
        echo "✅ Found libcuda.so.1 at: $LIBCUDA_PATH"
    else
        echo "❌ libcuda.so.1 not found anywhere"
        exit 1
    fi
fi

# Create environment setup script
echo "📝 Creating environment setup script..."
cat > setup_cuda_env.sh << EOF
#!/bin/bash
# CUDA Environment Setup Script
# Run this before training: source setup_cuda_env.sh

export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export NUMBA_CUDA_DRIVER=$LIBCUDA_PATH

# Reduce GPU memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

echo "✅ CUDA environment configured"
echo "CUDA_HOME: \$CUDA_HOME"
echo "NUMBA_CUDA_DRIVER: \$NUMBA_CUDA_DRIVER"
EOF

chmod +x setup_cuda_env.sh

# Create CPU-only setup script as well
echo "📝 Creating CPU-only environment setup script..."
cat > setup_cpu_env.sh << EOF
#!/bin/bash
# CPU-Only Environment Setup Script
# Run this for CPU-only training: source setup_cpu_env.sh

export CUDA_VISIBLE_DEVICES=""
export CUDF_BACKEND="cpu"
export RAPIDS_NO_INITIALIZE="1"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "✅ CPU-only environment configured"
echo "CUDA disabled for CPU-only training"
EOF

chmod +x setup_cpu_env.sh

echo "✅ Created setup_cuda_env.sh"
echo "✅ Created setup_cpu_env.sh"
echo ""
echo "🚀 Next steps:"
echo "For GPU training:"
echo "1. Run: source setup_cuda_env.sh"
echo "2. Run: python run_4gpu_1000rows.py"
echo ""
echo "For CPU-only training:"
echo "1. Run: source setup_cpu_env.sh"  
echo "2. Run: python run_4gpu_1000rows.py"
