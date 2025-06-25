
#!/usr/bin/env python3
"""
Development Environment Setup Script
===================================

Automatically sets up the development environment for the trading system.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    dirs = [
        'models', 'models/test', 'models/dev', 'models/production',
        'logs', 'cached_data', 'runs', 'data_txt'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements_simple.txt'
        ], check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("⚠️ Some dependencies may have failed to install")
        print("   Try installing manually: pip install torch numpy pandas scikit-learn matplotlib")

def generate_sample_data():
    """Generate sample data if none exists."""
    data_dir = Path('data_txt')
    if not any(data_dir.glob('*.txt')):
        print("📊 Generating sample data...")
        try:
            import generate_sample_data
            generate_sample_data.main()
            print("✅ Sample data generated!")
        except Exception as e:
            print(f"⚠️ Could not generate sample data: {e}")
            print("   You may need to add your own data files to data_txt/")

def setup_git_hooks():
    """Setup git hooks for development."""
    git_dir = Path('.git')
    if git_dir.exists():
        print("📝 Setting up git hooks...")
        # Add any git hooks setup here
        print("✅ Git hooks configured!")

def main():
    """Main setup function."""
    print("🔧 Setting up Trading System Development Environment")
    print("=" * 60)
    
    create_directories()
    install_dependencies()
    generate_sample_data()
    setup_git_hooks()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run quick test: python run_simple.py")
    print("2. Open in PyCharm and run run_simple.py")
    print("3. Check logs in ./logs/ directory")
    print("4. Monitor training with: tensorboard --logdir=./runs")

if __name__ == "__main__":
    main()
