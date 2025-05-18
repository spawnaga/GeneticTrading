# ga_policy_evolution.py

import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.multiprocessing import Pool, cpu_count, set_start_method
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Use 'spawn' to avoid CUDA issues when forking
# ──────────────────────────────────────────────────────────────────────────────
try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass


class PolicyNetwork(nn.Module):
    """
    A simple 3-layer MLP policy network for GA evolution.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        ).to(self.device)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

    def act(self, state):
        state = state.to(self.device)
        with torch.no_grad():
            logits = self.forward(state)
        return int(torch.argmax(logits, dim=-1).item())

    def get_params(self):
        return np.concatenate([p.cpu().data.numpy().ravel() for p in self.parameters()])

    def set_params(self, vector):
        vec = np.array(vector, dtype=np.float32)
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            chunk = vec[idx:idx + numel].reshape(p.shape)
            p.data.copy_(torch.from_numpy(chunk).to(self.device))
            idx += numel

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=self.device))
            print(f"[GA] Loaded existing model from {path}")
        else:
            print(f"[GA] No model found at {path}, starting fresh.")


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
        return step_ret  # obs, reward, done, info
    else:
        raise ValueError(f"Unexpected step return length: {len(step_ret)}")


def evaluate_fitness(param_vector, env, device="cpu"):
    """
    Roll out one episode with given parameters, return total scaled reward.
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
    reset_ret = env.reset()
    obs = _unpack_reset(reset_ret)

    done = False
    while not done:
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy.device)
        action = policy.act(state_tensor)

        step_ret = env.step(action)
        obs, reward, done, info = _unpack_step(step_ret)

        total_reward += float(reward) * 10.0  # scale up for GA

    return total_reward


def parallel_gpu_eval(args):
    params, env, gpu_id = args
    return evaluate_fitness(params, env, device=f"cuda:{gpu_id}")


def run_ga_evolution(
    env,
    population_size=40,
    generations=20,
    tournament_size=7,
    mutation_rate=0.8,
    mutation_scale=1.0,
    num_workers=None,
    device="cpu",
    model_save_path="ga_policy_model.pth"
):
    """
    Genetic Algorithm main loop with:
      - rank-0 only progress bar (left on-screen at end)
      - in-place tqdm updates via set_postfix
      - TensorBoard logging
      - elitism, tournament selection, crossover, adaptive mutation
      - per-generation print only on non-bar ranks
    """
    import os, sys, time
    import numpy as np
    import torch
    from tqdm import tqdm
    from torch.multiprocessing import Pool, cpu_count
    from torch.utils.tensorboard import SummaryWriter

    # 1) Verify env support
    import gymnasium as gym
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("GA policy only supports Box observation spaces")

    # 2) Determine rank and whether to show bar
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    disable_bar = (local_rank != 0)

    # 3) Create in-place tqdm bar on stdout, leave on screen when done
    bar = tqdm(
        range(generations),
        desc="[GA gens]",
        file=sys.stdout,
        disable=disable_bar,
        dynamic_ncols=True,
        leave=True,
    )

    # 4) Prepare network, population, and TensorBoard writer
    from ga_policy_evolution import PolicyNetwork, evaluate_fitness, parallel_gpu_eval
    input_dim  = int(np.prod(env.observation_space.shape))
    output_dim = env.action_space.n

    writer = SummaryWriter(log_dir="runs/ga_experiment", flush_secs=1)
    base_policy = PolicyNetwork(input_dim, 64, output_dim, device=device)
    base_policy.load_model(model_save_path)
    param_size = len(base_policy.get_params())

    population = [
        np.array(PolicyNetwork(input_dim, 64, output_dim, device=device)
                 .get_params(), dtype=np.float32)
        for _ in range(population_size)
    ]

    # 5) Initial fitness logging
    init_fits = [evaluate_fitness(ind, env, device) for ind in population]
    print(f"[GA DEBUG] Init fitness → "
          f"min={min(init_fits):.4f}, avg={np.mean(init_fits):.4f}, max={max(init_fits):.4f}")
    writer.add_scalar("GA/InitMinFitness", min(init_fits), 0)
    writer.add_scalar("GA/InitAvgFitness", np.mean(init_fits), 0)
    writer.add_scalar("GA/InitMaxFitness", max(init_fits), 0)

    best_fitness = -float("inf")
    best_params  = None
    stagnation   = 0

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    gpu_count = torch.cuda.device_count() if str(device).startswith("cuda") else 0

    start_time = time.time()

    try:
        for gen in bar:
            # ─── Evaluate population ─────────────────────────────────────────
            if gpu_count > 0:
                with Pool(gpu_count) as pool:
                    args = [(ind, env, i % gpu_count) for i, ind in enumerate(population)]
                    fitnesses = pool.map(parallel_gpu_eval, args)
            elif num_workers > 1:
                with Pool(num_workers) as pool:
                    args = [(ind, env, device) for ind in population]
                    fitnesses = pool.starmap(evaluate_fitness, args)
            else:
                fitnesses = [evaluate_fitness(ind, env, device) for ind in population]

            # ─── Compute stats ────────────────────────────────────────────────
            gen_best  = float(np.max(fitnesses))
            gen_avg   = float(np.mean(fitnesses))
            param_std = float(np.std(np.stack(population)))
            elapsed   = time.time() - start_time

            # ─── TensorBoard logging ─────────────────────────────────────────
            writer.add_scalar("GA/BestFitness", gen_best, gen)
            writer.add_scalar("GA/AvgFitness",   gen_avg,  gen)
            writer.add_histogram("GA/FitnessDist", np.array(fitnesses), gen)
            writer.add_scalar("GA/ParamStd",    param_std, gen)
            writer.add_scalar("GA/ElapsedSeconds", elapsed, gen)
            writer.add_text("GA/Generation", f"Generation {gen+1}/{generations}", gen)
            if gen % 10 == 0:
                w = base_policy.net[0].weight.data.cpu().numpy()
                writer.add_histogram("GA/Layer0Weights", w, gen)
            writer.flush()

            # ─── Update tqdm postfix in-place ────────────────────────────────
            bar.set_postfix({
                "best": f"{gen_best:.1f}",
                "avg":  f"{gen_avg:.1f}",
                "std":  f"{param_std:.3f}"
            })

            # ─── Optional console print only on non-bar ranks ────────────────
            if disable_bar:
                print(f"[GA] Gen {gen}: best={gen_best:.2f}, avg={gen_avg:.2f}, "
                      f"all_time_best={best_fitness:.2f}, param_std={param_std:.4f}")

            # ─── Elitism & checkpoint ─────────────────────────────────────────
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_idx     = int(np.argmax(fitnesses))
                best_params  = population[best_idx].copy()
                base_policy.set_params(best_params)
                base_policy.save_model(model_save_path)
                stagnation = 0
            else:
                stagnation += 1

            # ─── Breed next generation ────────────────────────────────────────
            elite_n = max(1, int(0.1 * population_size))
            elites  = [population[i] for i in np.argsort(fitnesses)[-elite_n:]]
            new_pop = elites.copy()
            while len(new_pop) < population_size:
                idxs1 = np.random.choice(population_size, tournament_size, replace=False)
                p1    = population[idxs1[np.argmax([fitnesses[i] for i in idxs1])]]
                idxs2 = np.random.choice(population_size, tournament_size, replace=False)
                p2    = population[idxs2[np.argmax([fitnesses[i] for i in idxs2])]]
                alpha = np.random.rand()
                child = alpha * p1 + (1 - alpha) * p2

                mask  = np.random.rand(param_size) < mutation_rate
                child[mask] += np.random.randn(int(mask.sum())) * (
                    mutation_scale * (1 + 0.1 * stagnation)
                )
                new_pop.append(child)

            population = new_pop

    finally:
        writer.close()

    # ─── Return best agent ────────────────────────────────────────────────
    final_agent = PolicyNetwork(input_dim, 64, output_dim, device=device)
    final_agent.set_params(best_params)
    return final_agent, best_fitness, None, None
