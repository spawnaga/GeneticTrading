import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import os
import cudf

from data_preprocessing import create_environment_data
from trading_environment import TradingEnvironment
from ga_policy_evolution import run_ga_evolution, PolicyNetwork
from policy_gradient_methods import PPOTrainer

def compute_performance_metrics(balance_history):
    steps_per_year = 252 * 390
    total_return = (balance_history[-1] - balance_history[0]) / balance_history[0]
    years = len(balance_history) / steps_per_year
    cagr = (1 + total_return)**(1 / years) - 1 if years > 0 else 0
    returns = np.diff(balance_history) / balance_history[:-1]
    ann_mean = np.mean(returns) * steps_per_year
    ann_std = np.std(returns) * np.sqrt(steps_per_year)
    sharpe = ann_mean / (ann_std + 1e-8)
    running_max = np.maximum.accumulate(balance_history)
    mdd = np.max((running_max - balance_history) / running_max)
    return cagr, sharpe, mdd

def evaluate_agent(env, agent, steps):
    balance_history = []
    obs = env.reset()
    device = next(agent.parameters()).device
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    for _ in range(steps):
        if env.done:
            break

        with torch.no_grad():
            if hasattr(agent, 'act'):
                action = agent.act(obs_tensor)
            else:
                policy_logits, _ = agent.forward(obs_tensor)
                dist_ = torch.distributions.Categorical(logits=policy_logits)
                action = dist_.sample().item()

        obs, _, done, _ = env.step(action)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        balance_history.append(env.current_balance)

        if done:
            break

    return balance_history

def main():
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(dist.get_rank())
    device = torch.device(f"cuda:{dist.get_rank()}")

    data_folder = './data_txt'
    cache_folder = './cached_data'
    os.makedirs(cache_folder, exist_ok=True)

    train_data_path = f'{cache_folder}/train_data.parquet'
    test_data_path = f'{cache_folder}/test_data.parquet'

    # ONLY RANK 0 prepares and caches data
    if local_rank == 0:
        if not (os.path.exists(train_data_path) and os.path.exists(test_data_path)):
            train_data, test_data, scaler = create_environment_data(
                data_folder=data_folder,
                max_rows=10000,
                use_gpu=True,
                cache_folder=cache_folder
            )
            train_data.to_parquet(train_data_path)
            test_data.to_parquet(test_data_path)
            print("✅ Data cached successfully.")
        else:
            print("✅ Cached data already exists.")

    dist.barrier()

    # All ranks load cached data after synchronization
    train_data = cudf.read_parquet(train_data_path)
    test_data = cudf.read_parquet(test_data_path)

    train_env = TradingEnvironment(train_data, initial_balance=100000.0)
    test_env = TradingEnvironment(test_data, initial_balance=100000.0)

    ga_model_path = "ga_policy_model.pth"
    ga_agent = PolicyNetwork(train_env.observation_dim, 64, train_env.action_space, device=device)

    # GA training (rank 0 only)
    if local_rank == 0:
        if os.path.exists(ga_model_path):
            ga_agent.load_model(ga_model_path)
        else:
            ga_agent, best_fit = run_ga_evolution(
                train_env,
                population_size=20,
                generations=6,
                device=device,
                model_save_path=ga_model_path
            )

    dist.barrier()
    ga_agent.load_model(ga_model_path)

    if local_rank == 0:
        balance_history_ga = evaluate_agent(test_env, ga_agent, len(test_data))
        cagr_ga, sharpe_ga, mdd_ga = compute_performance_metrics(balance_history_ga)
        print(f"GA Results - CAGR: {cagr_ga:.4f}, Sharpe: {sharpe_ga:.4f}, MaxDD: {mdd_ga:.4f}")

    # PPO Setup (rank 0 only)
    ppo_model_path = "ppo_actor_critic_model.pth"
    ppo_trainer = PPOTrainer(train_env,
                             input_dim=train_env.observation_dim,
                             action_dim=train_env.action_space,
                             hidden_dim=64,
                             lr=3e-4,
                             gamma=0.99,
                             clip_epsilon=0.2,
                             update_epochs=10,
                             rollout_steps=500,
                             device=device,
                             model_save_path=ppo_model_path)

    if local_rank == 0:
        if os.path.exists(ppo_model_path):
            ppo_trainer.model.load_model(ppo_model_path)
        else:
            ppo_trainer.train(total_timesteps=100000)
            ppo_trainer.model.save_model(ppo_model_path)

    dist.barrier()
    ppo_trainer.model.load_model(ppo_model_path)

    if local_rank == 0:
        balance_history_ppo = evaluate_agent(test_env, ppo_trainer.model, len(test_data))
        cagr_ppo, sharpe_ppo, mdd_ppo = compute_performance_metrics(balance_history_ppo)
        print(f"PPO Results - CAGR: {cagr_ppo:.4f}, Sharpe: {sharpe_ppo:.4f}, MaxDD: {mdd_ppo:.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(balance_history_ga, label='GA Policy')
        plt.plot(balance_history_ppo, label='PPO Policy')
        plt.title('Equity Curves Comparison')
        plt.xlabel('Time Steps')
        plt.ylabel('Balance')
        plt.legend()
        plt.grid(True)
        plt.show()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
