
#!/usr/bin/env python
"""
Enhanced Hybrid Trading System Runner
====================================

Runs the revolutionary hybrid GA-PPO trading system with:
- Enhanced realistic trading environment
- Knowledge transfer between algorithms
- Real-time visualization
- Comprehensive performance tracking
"""

import os
import sys
import argparse
import logging
import threading
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_preprocessing import create_environment_data
from futures_env import FuturesEnv
from hybrid_ga_ppo_trainer import HybridGAPPOTrainer
from live_trading_visualizer import LiveTradingVisualizer
from utils import build_states_for_futures_env, setup_logging
import pickle

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced Hybrid GA-PPO Trading System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument('--data-folder', type=str, default='./data_txt',
                       help='Path to historical data folder')
    parser.add_argument('--cache-folder', type=str, default='./cached_data',
                       help='Path to cache folder')
    parser.add_argument('--max-rows', type=int, default=50000,
                       help='Maximum rows to process (0 for all)')
    
    # Training configuration
    parser.add_argument('--max-iterations', type=int, default=30,
                       help='Maximum training iterations')
    parser.add_argument('--ga-generations', type=int, default=15,
                       help='GA generations per iteration')
    parser.add_argument('--ppo-updates', type=int, default=8,
                       help='PPO updates per iteration')
    
    # Environment configuration
    parser.add_argument('--initial-capital', type=float, default=100000,
                       help='Initial trading capital')
    parser.add_argument('--max-position-size', type=int, default=5,
                       help='Maximum position size')
    parser.add_argument('--commission', type=float, default=2.50,
                       help='Commission per contract')
    
    # System configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--models-dir', type=str, default='./models/hybrid',
                       help='Directory to save models')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Visualization
    parser.add_argument('--enable-visualization', action='store_true',
                       help='Enable real-time visualization')
    parser.add_argument('--viz-update-interval', type=int, default=5,
                       help='Visualization update interval (seconds)')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
            logging.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            logging.info("Auto-selected CPU device")
    else:
        device = device_arg
        
    # Verify device works
    try:
        test_tensor = torch.tensor([1.0]).to(device)
        logging.info(f"✅ Device {device} verified and ready")
    except Exception as e:
        logging.error(f"❌ Device {device} failed: {e}")
        device = 'cpu'
        logging.info("Falling back to CPU")
        
    return device


def create_enhanced_environments(args) -> tuple:
    """Create enhanced training and testing environments."""
    logging.info("📊 Creating enhanced trading environments...")
    
    # Setup directories
    Path(args.cache_folder).mkdir(parents=True, exist_ok=True)
    
    # Load or create data
    train_path = os.path.join(args.cache_folder, "train_data.parquet")
    test_path = os.path.join(args.cache_folder, "test_data.parquet")
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        logging.info("Processing raw data...")
        train_df, test_df, scaler, segment_dict = create_environment_data(
            data_folder=args.data_folder,
            max_rows=args.max_rows,
            test_size=0.2,
            cache_folder=args.cache_folder
        )
        
        # Save processed data
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        with open(os.path.join(args.cache_folder, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(args.cache_folder, "segment_dict.pkl"), "wb") as f:
            pickle.dump(segment_dict, f)
            
        logging.info("✅ Data processing completed")
    else:
        logging.info("📂 Loading cached data...")
        import pandas as pd
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
    # Convert to states
    train_df.rename(columns={"return": "return_"}, inplace=True)
    test_df.rename(columns={"return": "return_"}, inplace=True)
    
    train_states = build_states_for_futures_env(train_df)
    test_states = build_states_for_futures_env(test_df)
    
    logging.info(f"📈 Created {len(train_states)} training states, {len(test_states)} test states")
    
    # Create enhanced environments
    train_env = FuturesEnv(
        states=train_states,
        initial_capital=args.initial_capital,
        max_position_size=args.max_position_size,
        commission_per_contract=args.commission,
        value_per_tick=12.5,  # NQ futures
        tick_size=0.25
    )
    
    test_env = FuturesEnv(
        states=test_states,
        initial_capital=args.initial_capital,
        max_position_size=args.max_position_size,
        commission_per_contract=args.commission * 0.5,  # Reduced test commission
        value_per_tick=12.5,
        tick_size=0.25
    )
    
    logging.info("✅ Enhanced environments created successfully")
    return train_env, test_env


def run_hybrid_training(args, train_env, test_env, device: str):
    """Run the hybrid GA-PPO training."""
    logging.info("🚀 Starting Revolutionary Hybrid GA-PPO Training")
    
    # Setup models directory
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    
    # Get environment dimensions
    input_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    
    logging.info(f"🔧 Environment: {input_dim} inputs, {action_dim} actions")
    
    # Create hybrid trainer
    trainer = HybridGAPPOTrainer(
        train_env=train_env,
        test_env=test_env,
        input_dim=input_dim,
        action_dim=action_dim,
        device=device,
        models_dir=args.models_dir
    )
    
    # Run training
    results = trainer.train_hybrid(
        max_iterations=args.max_iterations,
        ga_generations=args.ga_generations,
        ppo_updates=args.ppo_updates
    )
    
    # Save best model
    best_model_path = os.path.join(args.models_dir, "best_hybrid_model.pth")
    trainer.save_best_model(best_model_path)
    
    logging.info(f"✅ Training completed! Best performance: {results['best_performance']:.4f}")
    logging.info(f"💾 Best model saved to: {best_model_path}")
    
    return results


def run_visualization_thread(args):
    """Run visualization in separate thread."""
    try:
        visualizer = LiveTradingVisualizer(
            update_interval=args.viz_update_interval
        )
        visualizer.start_visualization()
    except Exception as e:
        logging.error(f"Visualization error: {e}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(0)  # Main process
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logging.info("🎯 Enhanced Hybrid Trading System Starting...")
    logging.info(f"Configuration: {vars(args)}")
    
    # Setup device
    device = setup_device(args.device)
    
    try:
        # Create environments
        train_env, test_env = create_enhanced_environments(args)
        
        # Start visualization if enabled
        viz_thread = None
        if args.enable_visualization:
            logging.info("🎨 Starting real-time visualization...")
            viz_thread = threading.Thread(
                target=run_visualization_thread,
                args=(args,),
                daemon=True
            )
            viz_thread.start()
        
        # Run hybrid training
        results = run_hybrid_training(args, train_env, test_env, device)
        
        # Print summary
        print("\n" + "="*60)
        print("🏆 TRAINING SUMMARY")
        print("="*60)
        print(f"Best Performance: {results['best_performance']:.4f}")
        print(f"Total Iterations: {args.max_iterations}")
        print(f"Models saved to: {args.models_dir}")
        
        # Performance history summary
        history = results['performance_history']
        if history['hybrid_scores']:
            print(f"\nPerformance Evolution:")
            print(f"  Initial: {history['hybrid_scores'][0]:.4f}")
            print(f"  Final:   {history['hybrid_scores'][-1]:.4f}")
            print(f"  Best:    {max(history['hybrid_scores']):.4f}")
            
        if history['algorithm_switches']:
            print(f"\nAlgorithm Switches: {len(history['algorithm_switches'])}")
            for switch in history['algorithm_switches'][-3:]:  # Last 3 switches
                print(f"  Iteration {switch['iteration']}: {switch['from']} → {switch['to']} ({switch['reason']})")
                
        print("\n✅ System completed successfully!")
        
        # Keep visualization running if enabled
        if args.enable_visualization and viz_thread:
            logging.info("Visualization continues running... Press Ctrl+C to exit")
            try:
                viz_thread.join()
            except KeyboardInterrupt:
                logging.info("Stopping...")
                
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    finally:
        logging.info("Cleaning up...")


if __name__ == "__main__":
    main()
