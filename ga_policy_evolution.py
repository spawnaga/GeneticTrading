#!/usr/bin/env python
"""
GA-based policy evolution that shares exactly the same network architecture
with PPO (ActorCriticNet), plus:

  - Hall-of-Fame archive & injection
  - Linear decay of mutation_rate & mutation_scale
  - Local gradient-based refinement via PPOTrainer
  - Enhanced TensorBoard logging
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


def _unpack_reset(reset_ret):
    """
    Handle both Gymnasium (obs, info) and custom obs-only resets.
    """
    if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
        return reset_ret[0]
    return reset_ret


def _unpack_step(step_ret):
    """
    Handle both:
      - Gymnasium 5-tuple: (obs, reward, done, truncated, info)
      - Custom 4-tuple:   (obs, reward, done, info)
    Returns: obs, reward, done, info
    """
    if len(step_ret) == 5:
        obs, reward, done, truncated, info = step_ret
        return obs, reward, (done or truncated), info
    elif len(step_ret) == 4:
        return step_ret
    else:
        raise ValueError(f"Unexpected step return length: {len(step_ret)}")


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


from utils import compute_performance_metrics


def evaluate_fitness(param_vector, env, device="cpu", metric="profit"):
    """
    Roll out one episode with given parameters and return a fitness value.

    Args:
        param_vector: Flattened network parameters.
        env: Evaluation environment.
        device: Torch device string.
        metric: "profit" for raw profit sum or "sharpe" for Sharpe-minus-drawdown.
    """
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("GA policy only supports Box observation spaces")

    policy = PolicyNetwork(
        input_dim=int(np.prod(env.observation_space.shape)),
        hidden_dim=64,
        output_dim=env.action_space.n,
        device=device
    )
    policy.set_params(param_vector)

    total_reward = 0.0
    profits, times = [], []
    obs = _unpack_reset(env.reset())
    done = False
    while not done:
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy.device)
        action = policy.act(state_tensor)
        obs, reward, done, info = _unpack_step(env.step(action))
        total_reward += float(reward)
        if metric == "sharpe":
            profits.append(float(reward))
            times.append(info.get("timestamp"))

    if metric == "sharpe":
        _, sharpe, mdd = compute_performance_metrics(profits, times)
        return float(sharpe - mdd)
    else:
        return total_reward


def parallel_gpu_eval(args):
    """Allow multiprocessing evaluation across multiple GPUs."""
    params, env, gpu_id, metric = args
    return evaluate_fitness(params, env, device=f"cuda:{gpu_id}", metric=metric)


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
    fitness_metric="profit"
):
    """
    Genetic Algorithm main loop with:
      - Hall-of-Fame archive and periodic re-injection
      - Linear decay of mutation parameters
      - Local refinement via PPOTrainer every `local_refinement_interval`
      - Enhanced TensorBoard logging

    Args:
        env: trading environment
        population_size: number of genomes per generation
        generations: number of GA iterations
        tournament_size: selection tournament size
        mutation_rate: initial mutation probability
        mutation_scale: mutation noise std
        hall_of_fame_size: number of top individuals kept
        inject_interval: how often to inject hall-of-famers
        local_refinement_interval: generations between PPO refinement
        n_local_updates: PPO updates during refinement
        num_workers: CPU workers for evaluation
        device: torch device string
        model_save_path: where to save the champion
        fitness_metric: "profit" or "sharpe" fitness evaluation
    """
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    disable_bar = (local_rank != 0)
    bar = tqdm(
        range(generations),
        desc="[GA gens]",
        file=sys.stdout,
        disable=disable_bar,
        dynamic_ncols=True,
        leave=True,
    )

    writer = SummaryWriter(log_dir="runs/ga_experiment", flush_secs=1)

    # set up base policy for sizing & loading
    input_dim   = int(np.prod(env.observation_space.shape))
    output_dim  = env.action_space.n
    base_policy = PolicyNetwork(input_dim, 64, output_dim, device=device)
    try:
        base_policy.load_model(model_save_path)
    except AttributeError:
        pass  # first run, no model exists yet
    param_size  = len(base_policy.get_params())

    # initialize population
    pop = [ base_policy.get_params().copy() for _ in range(population_size) ]
    hall_of_fame = []  # list of (fitness, params)

    # initial fitness logging
    inits = [ evaluate_fitness(ind, env, device, metric=fitness_metric) for ind in pop ]
    writer.add_scalar("GA/InitMinFitness",    np.min(inits),    0)
    writer.add_scalar("GA/InitAvgFitness",    np.mean(inits),   0)
    writer.add_scalar("GA/InitMedianFitness", np.median(inits), 0)
    writer.add_scalar("GA/InitMaxFitness",    np.max(inits),    0)

    best_fitness = -float("inf")
    best_params  = None
    stagnation   = 0

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    gpu_count = torch.cuda.device_count() if device.startswith("cuda") else 0

    start_time     = time.time()
    init_mut_rate  = mutation_rate
    init_mut_scale = mutation_scale

    try:
        for gen in bar:
            # linearly decay mutation
            mu_r = init_mut_rate  * (1 - gen / max(1, generations - 1))
            mu_s = init_mut_scale * (1 - gen / max(1, generations - 1))

            # evaluate
            if gpu_count > 0:
                with Pool(gpu_count) as p:
                    args = [(ind, env, i % gpu_count, fitness_metric) for i, ind in enumerate(pop)]
                    fits = p.map(parallel_gpu_eval, args)
            elif num_workers > 1:
                with Pool(num_workers) as p:
                    args = [(ind, env, device, fitness_metric) for ind in pop]
                    fits = p.starmap(evaluate_fitness, args)
            else:
                fits = [evaluate_fitness(ind, env, device, metric=fitness_metric) for ind in pop]

            # stats
            gen_min, gen_avg, gen_med, gen_max = (
                float(np.min(fits)),
                float(np.mean(fits)),
                float(np.median(fits)),
                float(np.max(fits)),
            )
            param_std = float(np.std(np.stack(pop)))
            elapsed   = time.time() - start_time

            # TensorBoard
            writer.add_scalar("GA/MinFitness",     gen_min, gen)
            writer.add_scalar("GA/AvgFitness",     gen_avg, gen)
            writer.add_scalar("GA/MedianFitness",  gen_med, gen)
            writer.add_scalar("GA/MaxFitness",     gen_max, gen)
            writer.add_histogram("GA/FitnessDist", np.array(fits), gen)
            writer.add_scalar("GA/ParamStd",       param_std, gen)
            writer.add_scalar("GA/ElapsedSeconds", elapsed, gen)
            writer.add_scalar("GA/Stagnation",     stagnation, gen)
            writer.add_scalar("GA/MutationRate",   mu_r, gen)
            writer.add_scalar("GA/MutationScale",  mu_s, gen)
            writer.add_text("GA/GenInfo",
                            f"Gen {gen+1}/{generations} | best={gen_max:.2f} avg={gen_avg:.2f} std={param_std:.3f} stag={stagnation}",
                            gen)
            writer.flush()

            # progress-bar postfix
            bar.set_postfix({
                "min": f"{gen_min:.1f}", "avg": f"{gen_avg:.1f}",
                "med": f"{gen_med:.1f}", "max": f"{gen_max:.1f}",
                "std": f"{param_std:.3f}", "stg": f"{stagnation}",
                "mu_r": f"{mu_r:.2f}",   "mu_s": f"{mu_s:.2f}"}
            )

            # maintain Hall of Fame
            for f,p in zip(fits, pop):
                hall_of_fame.append((f, p.copy()))
            hall_of_fame = sorted(hall_of_fame, key=lambda x: x[0], reverse=True)[:hall_of_fame_size]

            # local PPO refinement
            if gen % local_refinement_interval == 0 and best_params is not None:
                champ = PolicyNetwork(input_dim, 64, output_dim, device=device)
                champ.set_params(best_params)
                ppo = PPOTrainer(env, input_dim, output_dim, device=device, model_save_path=None,)
                ppo.model.load_state_dict(champ.state_dict())
                for _ in range(n_local_updates):
                    ppo.train_step()
                best_params = np.concatenate([p.cpu().data.numpy().ravel()
                                               for p in ppo.model.parameters()])

            # elitism & checkpoint
            if gen_max > best_fitness:
                best_fitness = gen_max
                best_params  = pop[int(np.argmax(fits))].copy()
                base_policy.set_params(best_params)
                base_policy.save_model(model_save_path)
                stagnation = 0
            else:
                stagnation += 1

            # Hall of Fame injection
            if gen % inject_interval == 0 and hall_of_fame:
                for _ in range(min(len(hall_of_fame), population_size // 10)):
                    idx = np.random.randint(population_size)
                    pop[idx] = hall_of_fame[np.random.randint(len(hall_of_fame))][1].copy()

            # breed next gen
            elite_n = max(1, int(0.1 * population_size))
            elites  = [pop[i] for i in np.argsort(fits)[-elite_n:]]
            new_pop = elites.copy()
            while len(new_pop) < population_size:
                i1, i2 = np.random.choice(population_size, tournament_size, replace=False), None
                p1 = pop[i1[np.argmax([fits[i] for i in i1])]]
                i2 = np.random.choice(population_size, tournament_size, replace=False)
                p2 = pop[i2[np.argmax([fits[i] for i in i2])]]
                alpha = np.random.rand()
                child = alpha * p1 + (1 - alpha) * p2
                mask  = np.random.rand(param_size) < mu_r
                child[mask] += np.random.randn(int(mask.sum())) * mu_s
                new_pop.append(child)
            pop = new_pop

    finally:
        writer.close()

    # return best agent
    champ = PolicyNetwork(input_dim, 64, output_dim, device=device)
    champ.set_params(best_params)
    return champ, best_fitness, None, None
