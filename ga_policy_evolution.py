#!/usr/bin/env python
"""
GA-based policy evolution that shares exactly the same network architecture
with PPO (ActorCriticNet), plus:

  - Hall-of-Fame archive & injection
  - Linear decay of mutation_rate & mutation_scale
  - Local gradient-based refinement via PPOTrainer
  - Enhanced TensorBoard logging (including full GA config)
  - Clean tqdm progress-bar output with tqdm.write()
"""

import os
import sys
import time
import gymnasium as gym
import numpy as np
import torch
from torch.multiprocessing import Pool, cpu_count, set_start_method
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

from policy_gradient_methods import PPOTrainer, ActorCriticNet
from utils import compute_performance_metrics, _unpack_reset, _unpack_step, cleanup_tensorboard_runs

# ──────────────────────────────────────────────────────────────────────────────
# GA logger → always stderr, so tqdm bar on stdout stays intact
# ──────────────────────────────────────────────────────────────────────────────
ga_logger = logging.getLogger("ga_policy")
ga_logger.setLevel(logging.INFO)
if not ga_logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [GA] %(message)s"))
    ga_logger.addHandler(handler)

# ──────────────────────────────────────────────────────────────────────────────
# Use 'spawn' to avoid CUDA issues when forking
# ──────────────────────────────────────────────────────────────────────────────
try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass


class PolicyNetwork(ActorCriticNet):
    """
    GA version of the policy network; inherits the shared base and policy_head
    from ActorCriticNet. Only policy_head is used; value_head is ignored.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, device="cpu"):
        super().__init__(input_dim, hidden_dim, action_dim=output_dim, device=device)

    def forward(self, x):
        logits, _ = super().forward(x)
        return logits

    def act(self, state):
        state = state.to(self.device)
        with torch.no_grad():
            logits = self.forward(state)
            return int(torch.argmax(logits, dim=-1).item())

    def get_params(self):
        """
        Flatten base + policy_head parameters into one vector.
        """
        parts = []
        for p in self.base.parameters():
            parts.append(p.cpu().data.numpy().ravel())
        for p in self.policy_head.parameters():
            parts.append(p.cpu().data.numpy().ravel())
        return np.concatenate(parts)

    def set_params(self, vector):
        """
        Unflatten and copy into base + policy_head.
        """
        vec = np.array(vector, dtype=np.float32)
        idx = 0
        for p in list(self.base.parameters()) + list(self.policy_head.parameters()):
            numel = p.numel()
            chunk = vec[idx:idx + numel].reshape(p.shape)
            p.data.copy_(torch.from_numpy(chunk).to(self.device))
            idx += numel

    def save_model(self, path):
        """
        Save just the policy weights (base + policy_head).
        Use tqdm.write() so it doesn't break the progress bar.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        tqdm.write(f"[GA] Saved GA model to {path}")

    def load_model(self, path):
        """
        Load model weights if the file exists, otherwise keep random initialization.
        """
        if os.path.exists(path):
            try:
                saved_state = torch.load(path, map_location=self.device)

                # Check if dimensions match
                if 'base.0.weight' in saved_state:
                    saved_input_dim = saved_state['base.0.weight'].shape[1]
                    current_input_dim = self.base[0].weight.shape[1]

                    if saved_input_dim != current_input_dim:
                        tqdm.write(f"[GA] Dimension mismatch: saved model expects {saved_input_dim}, current model has {current_input_dim}")
                        tqdm.write(f"[GA] Starting from scratch due to incompatible checkpoint at {path}")
                        return

                self.load_state_dict(saved_state)
                tqdm.write(f"[GA] Loaded GA model from {path}")
            except Exception as e:
                tqdm.write(f"[GA] Failed to load model from {path}: {e}")
                tqdm.write("[GA] Starting from scratch")
        else:
            tqdm.write(f"[GA] No model file at {path}, starting from scratch")


def evaluate_fitness(param_vector, env, device="cpu", metric="comprehensive"):
    """
    Simplified fitness evaluation focused on stability and consistency.
    """
    import warnings
    warnings.filterwarnings("ignore", message="cudf import failed")

    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("GA policy only supports Box observation spaces")

    policy = PolicyNetwork(
        input_dim=int(np.prod(env.observation_space.shape)),
        hidden_dim=64,
        output_dim=env.action_space.n,
        device=device
    )
    policy.set_params(param_vector)

    # Run multiple episodes for stability
    episode_returns = []
    all_profits = []

    for episode in range(3):  # Multiple episodes for more stable evaluation
        obs = _unpack_reset(env.reset())
        done = False
        episode_reward = 0.0
        episode_profits = []

        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy.device)
            action = policy.act(state_tensor)
            obs, reward, done, info = _unpack_step(env.step(action))

            episode_reward += float(reward)
            episode_profits.append(float(reward))

        episode_returns.append(episode_reward)
        all_profits.extend(episode_profits)

    # Calculate stable metrics
    mean_return = np.mean(episode_returns)
    return_stability = 1.0 / (1.0 + np.std(episode_returns))  # Higher is better

    # Trade analysis
    total_trades = 0
    profitable_trades = 0
    if hasattr(env, 'trades') and len(env.trades) > 0:
        trade_profits = [trade[4] for trade in env.trades]
        total_trades = len(trade_profits)
        profitable_trades = sum(1 for p in trade_profits if p > 0)

    #```python
    # Simple, stable fitness
    fitness = (
        mean_return * 0.6 +  # Primary: average return
        return_stability * 50.0 * 0.3 +  # Stability bonus
        (profitable_trades / max(total_trades, 1)) * 20.0 * 0.1  # Win rate bonus
    )

    return float(fitness)


def parallel_gpu_eval(args):
    """
    Allow multiprocessing evaluation across multiple GPUs.
    """
    params, env, gpu_id, metric = args
    return evaluate_fitness(params, env,
                            device=f"cuda:{gpu_id}",
                            metric=metric)


def _log_ga_visualizations(writer, fits, population, generation, gen_max, gen_avg, best_fitness):
    """Create comprehensive visualizations for GA progress"""
    import matplotlib
    matplotlib.use('Agg')  # Ensure non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Fitness Evolution Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Fitness distribution
    ax1.hist(fits, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    ax1.axvline(gen_max, color='red', linestyle='--', label=f'Max: {gen_max:.4f}')
    ax1.axvline(gen_avg, color='green', linestyle='--', label=f'Avg: {gen_avg:.4f}')
    ax1.axvline(np.median(fits), color='orange', linestyle='--', label=f'Median: {np.median(fits):.4f}')
    ax1.set_title(f'Fitness Distribution - Generation {generation}')
    ax1.set_xlabel('Fitness Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Fitness box plot for outlier analysis
    ax2.boxplot(fits, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax2.set_title('Fitness Distribution Box Plot')
    ax2.set_ylabel('Fitness Value')
    ax2.grid(True, alpha=0.3)

    # Population parameter diversity (sample of parameters)
    if len(population) > 0:
        # Sample first 100 parameters for visualization
        param_sample = np.array([ind[:100] for ind in population[:min(20, len(population))]])
        im = ax3.imshow(param_sample, aspect='auto', cmap='viridis')
        ax3.set_title('Population Parameter Diversity (Sample)')
        ax3.set_xlabel('Parameter Index (First 100)')
        ax3.set_ylabel('Individual Index (First 20)')
        plt.colorbar(im, ax=ax3, label='Parameter Value')

    # Performance metrics summary
    metrics_text = f"""
    Generation: {generation}
    Population Size: {len(population)}

    Fitness Statistics:
    Max: {gen_max:.6f}
    Mean: {gen_avg:.6f}
    Median: {np.median(fits):.6f}
    Std: {np.std(fits):.6f}

    Diversity:
    Parameter Std: {np.std(np.stack(population)):.6f}
    Fitness Range: {gen_max - np.min(fits):.6f}

    Best Ever: {best_fitness:.6f}
    Improvement: {gen_max - best_fitness if gen_max > best_fitness else 0:.6f}
    """

    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    ax4.set_title('Generation Summary')
    ax4.axis('off')

    plt.tight_layout()
    writer.add_figure("GA/Visualizations/Generation_Analysis", fig, generation)
    plt.close(fig)

    # 2. Advanced fitness analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Fitness vs rank
    sorted_fits = np.sort(fits)[::-1]  # Descending order
    ranks = np.arange(1, len(sorted_fits) + 1)
    ax1.plot(ranks, sorted_fits, 'o-', alpha=0.7)
    ax1.set_title('Fitness by Rank')
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Fitness')
    ax1.grid(True, alpha=0.3)

    # Selection pressure analysis
    total_fitness = np.sum(fits)
    selection_probs = fits / (total_fitness + 1e-8)
    sorted_probs = np.sort(selection_probs)[::-1]
    ax2.plot(ranks, sorted_probs, 'o-', alpha=0.7, color='red')
    ax2.set_title('Selection Probability by Rank')
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Selection Probability')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    writer.add_figure("GA/Visualizations/Selection_Analysis", fig, generation)
    plt.close(fig)


def run_ga_evolution(
    env,
    population_size=40,
    generations=20,
    tournament_size=7,
    mutation_rate=0.8,
    mutation_scale=1.0,
    hall_of_fame_size=5,
    inject_interval=10,
    local_refinement_interval=10,
    n_local_updates=5,
    num_workers=None,
    device="cpu",
    model_save_path="ga_policy_model.pth",
    fitness_metric: str = "comprehensive"
):
    """
    Genetic Algorithm main loop with:
      - Hall-of-Fame archive and periodic re-injection
      - Linear decay of mutation parameters
      - Local refinement via PPOTrainer
      - Enhanced TensorBoard logging (including GA config)
      - Enhanced TensorBoard logging (including GA config)
    """
    import os  # Ensure os is available in function scope
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    disable_bar = (local_rank != 0)
    bar = tqdm(
        range(generations),
        desc="[GA gens]",
        file=sys.stdout,
        disable=disable_bar,
        ncols=120,
        leave=False,
        bar_format="{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    # Create TensorBoard directory first, then cleanup old runs
    tb_log_dir = "./runs/ga_experiment"
    os.makedirs(tb_log_dir, exist_ok=True)

    if local_rank == 0:
        cleanup_tensorboard_runs("./runs", keep_latest=2)  # Keep more runs to avoid conflicts
        # Ensure the specific directory still exists after cleanup
        os.makedirs(tb_log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_log_dir, flush_secs=1)
    # Log configuration with error handling
    if writer:
        try:
            writer.add_text(
                "GA/Config",
                f"pop={population_size}, gens={generations}, tour={tournament_size}, "
                f"mut_rate={mutation_rate}, mut_scale={mutation_scale}, HOF={hall_of_fame_size}, "
                f"inject_int={inject_interval}, refine_int={local_refinement_interval}, "
                f"n_refine={n_local_updates}, metric={fitness_metric}",
                0
            )
        except Exception as e:
            ga_logger.warning(f"TensorBoard config logging failed: {e}")
            writer = None

    # set up base policy for sizing & optional checkpoint load
    input_dim = int(np.prod(env.observation_space.shape))
    output_dim = env.action_space.n
    base_policy = PolicyNetwork(input_dim, 64, output_dim, device=device)
    base_policy.load_model(model_save_path)  # logs internally if missing
    param_size = len(base_policy.get_params())

    # initialize population & hall-of-fame with better diversity
    pop = []
    base_params = base_policy.get_params()

    # Add base policy
    pop.append(base_params.copy())

    # Add variations with different noise levels
    for i in range(population_size - 1):
        noise_scale = 0.1 * (1 + i / population_size)  # Increasing noise
        variant = base_params + np.random.normal(0, noise_scale, size=base_params.shape)
        pop.append(variant)

    hall_of_fame = []

    # initial fitness logging
    inits = [evaluate_fitness(ind, env, device, metric=fitness_metric) for ind in pop]
    writer.add_scalar("GA/InitMinFitness", np.min(inits), 0)
    writer.add_scalar("GA/InitAvgFitness", np.mean(inits), 0)
    writer.add_scalar("GA/InitMedianFitness", np.median(inits), 0)
    writer.add_scalar("GA/InitMaxFitness", np.max(inits), 0)

    best_fitness, best_params = -float("inf"), None
    stagnation = 0

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    gpu_count = torch.cuda.device_count() if str(device).startswith("cuda") else 0

    start_time = time.time()
    init_mut_rate, init_mut_scale = mutation_rate, mutation_scale

    try:
        for gen in bar:
            # decay mutation parameters
            mu_r = init_mut_rate * (1 - gen / max(1, generations - 1))
            mu_s = init_mut_scale * (1 - gen / max(1, generations - 1))

            # evaluate population (prefer single-threaded for stability)
            if gpu_count > 1 and num_workers > 4:  # Only use multiprocessing with sufficient resources
                try:
                    with Pool(min(4, gpu_count)) as p:  # Limit workers to prevent resource exhaustion
                        args = [
                            (ind, env, i % gpu_count, fitness_metric)
                            for i, ind in enumerate(pop)
                        ]
                        fits = p.map(parallel_gpu_eval, args)
                except Exception as e:
                    tqdm.write(f"[GA] Multiprocessing failed ({e}), falling back to single-thread")
                    fits = [
                        evaluate_fitness(ind, env, device, metric=fitness_metric)
                        for ind in pop
                    ]
            else:
                # Single-threaded evaluation (more stable in Replit)
                fits = [
                    evaluate_fitness(ind, env, device, metric=fitness_metric)
                    for ind in pop
                ]

            # compute stats
            gen_min = float(np.min(fits))
            gen_avg = float(np.mean(fits))
            gen_med = float(np.median(fits))
            gen_max = float(np.max(fits))
            param_std = float(np.std(np.stack(pop)))
            elapsed = time.time() - start_time

            # Enhanced TensorBoard logging with better error handling
            try:
                if writer and writer.file_writer:
                    writer.add_scalar("GA/Fitness/Min", gen_min, gen)
                    writer.add_scalar("GA/Fitness/Average", gen_avg, gen)
                    writer.add_scalar("GA/Fitness/Median", gen_med, gen)
                    writer.add_scalar("GA/Fitness/Max", gen_max, gen)
                    writer.add_scalar("GA/Fitness/Range", gen_max - gen_min, gen)
                    writer.add_scalar("GA/Fitness/IQR", np.percentile(fits, 75) - np.percentile(fits, 25), gen)
            except Exception as tb_error:
                ga_logger.warning(f"TensorBoard logging failed for generation {gen}: {tb_error}")
                writer = None

            # Population diversity metrics
            try:
                if writer and writer.file_writer:
                    writer.add_scalar("GA/Diversity/Parameter_Std", param_std, gen)
                    writer.add_scalar("GA/Diversity/Fitness_Std", float(np.std(fits)), gen)
                    writer.add_scalar("GA/Diversity/Fitness_CV", float(np.std(fits) / (np.mean(fits) + 1e-8)), gen)

                    # Evolution dynamics
                    writer.add_scalar("GA/Evolution/Stagnation", stagnation, gen)
                    writer.add_scalar("GA/Evolution/Mutation_Rate", mu_r, gen)
                    writer.add_scalar("GA/Evolution/Mutation_Scale", mu_s, gen)
                    writer.add_scalar("GA/Evolution/Elite_Fitness", gen_max, gen)
            except Exception as tb_error:
                ga_logger.warning(f"TensorBoard diversity/evolution logging failed for generation {gen}: {tb_error}")
                writer = None

            # Performance tracking
            try:
                if writer and writer.file_writer:
                    writer.add_scalar("GA/Performance/Elapsed_Time", elapsed, gen)
                    writer.add_scalar("GA/Performance/Generation_Rate", gen / (elapsed + 1e-8), gen)

                    # Histograms and distributions
                    writer.add_histogram("GA/Fitness_Distribution", np.array(fits), gen)

                    # Calculate fitness improvement
                    if gen > 0:
                        improvement = gen_max - best_fitness if gen_max > best_fitness else 0
                        writer.add_scalar("GA/Evolution/Fitness_Improvement", improvement, gen)

                    # Log percentiles for detailed analysis
                    percentiles = [10, 25, 75, 90]
                    for p in percentiles:
                        writer.add_scalar(f"GA/Fitness/Percentile_{p}", np.percentile(fits, p), gen)

                    # Detailed text logging
                    writer.add_text(
                        "GA/Generation_Summary",
                        f"Gen {gen+1}/{generations} | Best: {gen_max:.4f} | Avg: {gen_avg:.4f} | "
                        f"Std: {param_std:.4f} | Stagnation: {stagnation} | "
                        f"Mut Rate: {mu_r:.3f} | Mut Scale: {mu_s:.3f} | "
                        f"Improvement: {gen_max - best_fitness if gen_max > best_fitness else 0:.4f}",
                        gen
                    )
                    writer.flush()
            except Exception as tb_error:
                ga_logger.warning(f"TensorBoard performance/text logging failed for generation {gen}: {tb_error}")
                writer = None

            # Create detailed visualizations less frequently to prevent threading issues
            if gen % 20 == 0 and gen > 0:  # Less frequent, skip first generation
                try:
                    if writer and writer.file_writer:
                        _log_ga_visualizations(writer, fits, pop, gen, gen_max, gen_avg, best_fitness)
                except Exception as e:
                    tqdm.write(f"[GA] Visualization failed (gen {gen}): {e}")
                    # Continue without visualization

            # update tqdm postfix
            bar.set_description(f"[GA gens] avg={np.mean(fits):.1f}, max={np.max(fits):.1f}, stg={stagnation}")

            # update hall-of-fame
            for f, p in zip(fits, pop):
                hall_of_fame.append((f, p.copy()))
            hall_of_fame = sorted(hall_of_fame, key=lambda x: x[0], reverse=True)[:hall_of_fame_size]

            # PPO-based local refinement
            if gen % local_refinement_interval == 0 and best_params is not None:
                champ = PolicyNetwork(input_dim, 64, output_dim, device=device)
                champ.set_params(best_params)
                ppo = PPOTrainer(env, input_dim, output_dim, device=device)
                ppo.model.load_state_dict(champ.state_dict())
                for _ in range(n_local_updates):
                    ppo.train_step()
                best_params = np.concatenate([
                    p.cpu().data.numpy().ravel()
                    for p in ppo.model.parameters()
                ])

            # elitism & checkpoint
            if gen_max > best_fitness:
                best_fitness = gen_max
                best_params = pop[int(np.argmax(fits))].copy()
                base_policy.set_params(best_params)
                base_policy.save_model(model_save_path)
                stagnation = 0
            else:
                stagnation += 1

            # inject Hall-of-Fame members
            if gen % inject_interval == 0 and hall_of_fame:
                for _ in range(min(len(hall_of_fame), population_size // 10)):
                    idx = np.random.randint(population_size)
                    pop[idx] = hall_of_fame[np.random.randint(len(hall_of_fame))][1].copy()

            # breed next generation with improved strategy
            elite_n = max(2, int(0.15 * population_size))  # Keep more elites
            elite_indices = np.argsort(fits)[-elite_n:]
            elites = [pop[i] for i in elite_indices]
            new_pop = elites.copy()

            # Add diversity injection
            if stagnation > 5:  # If stagnating, inject more diversity
                diversity_boost = min(population_size // 4, 10)
                for _ in range(diversity_boost):
                    base_params = base_policy.get_params()
                    diverse_variant = base_params + np.random.normal(0, mu_s * 2, size=base_params.shape)
                    new_pop.append(diverse_variant)

            while len(new_pop) < population_size:
                # Improved tournament selection
                if np.random.rand() < 0.7:  # 70% tournament selection
                    idxs1 = np.random.choice(population_size, tournament_size, replace=False)
                    p1 = pop[idxs1[np.argmax([fits[i] for i in idxs1])]]
                    idxs2 = np.random.choice(population_size, tournament_size, replace=False)
                    p2 = pop[idxs2[np.argmax([fits[i] for i in idxs2])]]
                else:  # 30% random selection for diversity
                    p1 = pop[np.random.randint(population_size)]
                    p2 = pop[np.random.randint(population_size)]

                # Adaptive crossover
                alpha = np.random.beta(2, 2)  # Beta distribution for smoother blending
                child = alpha * p1 + (1 - alpha) * p2

                # Adaptive mutation
                adaptive_mu_r = mu_r * (1 + stagnation * 0.1)  # Increase mutation if stagnating
                mask = np.random.rand(param_size) < adaptive_mu_r
                mutation_strength = mu_s * (1 + np.random.exponential(0.5))  # Variable mutation strength
                child[mask] += np.random.randn(int(mask.sum())) * mutation_strength

                new_pop.append(child)
            pop = new_pop

    finally:
        try:
            if writer:
                writer.flush()
                writer.close()
        except Exception as e:
            ga_logger.warning(f"Error closing TensorBoard writer: {e}")

    # build and return final champion
    champ = PolicyNetwork(input_dim, 64, output_dim, device=device)
    champ.set_params(best_params)
    return champ, best_fitness, None, None