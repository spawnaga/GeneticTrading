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

# ──────────────────────────────────────────────────────────────────────────────
# Reuse the ActorCriticNet so GA & PPO share exactly the same architecture.
# ──────────────────────────────────────────────────────────────────────────────
from policy_gradient_methods import ActorCriticNet

class PolicyNetwork(ActorCriticNet):
    """
    GA version of the policy net; inherits the shared-base + policy_head
    from ActorCriticNet. We simply never use the value_head here.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, device="cpu"):
        # note: output_dim → action_dim in ActorCriticNet
        super().__init__(input_dim, hidden_dim, action_dim=output_dim, device=device)

    def forward(self, x):
        # returns only the policy logits, ignores the value head
        logits, _ = super().forward(x)
        return logits

    def act(self, state):
        """
        State is a torch.Tensor. Returns discrete action.
        """
        state = state.to(self.device)
        with torch.no_grad():
            logits = self.forward(state)
        return int(torch.argmax(logits, dim=-1).item())

    def get_params(self):
        """
        Flatten base + policy_head parameters into one vector.
        """
        params = []
        # base layers
        for p in self.base.parameters():
            params.append(p.cpu().data.numpy().ravel())
        # policy head
        for p in self.policy_head.parameters():
            params.append(p.cpu().data.numpy().ravel())
        return np.concatenate(params)

    def set_params(self, vector):
        """
        Unflatten and copy into base + policy_head.
        """
        vec = np.array(vector, dtype=np.float32)
        idx = 0
        for p in list(self.base.parameters()) + list(self.policy_head.parameters()):
            numel = p.numel()
            chunk = vec[idx : idx + numel].reshape(p.shape)
            p.data.copy_(torch.from_numpy(chunk).to(self.device))
            idx += numel

    def save_model(self, path):
        """
        Save just the policy weights (base + policy_head).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """
        Load weights if they exist, else random init.
        """
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
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = policy.act(state_tensor)

        step_ret = env.step(action)
        obs, reward, done, info = _unpack_step(step_ret)

        total_reward += float(reward) * 10.0  # scale up for GA

    return total_reward


def parallel_gpu_eval(args):
    """
    Allow multiprocessing across GPUs.
    """
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
      - shared ActorCriticNet architecture for GA & PPO
      - TensorBoard logging of min/avg/median/max fitness + distribution + parameter std + stagnation
      - full original GA machinery, plus now perfect symmetry with PPO’s load_state_dict
    """
    local_rank   = int(os.environ.get("LOCAL_RANK", 0))
    disable_bar  = (local_rank != 0)
    bar = tqdm(
        range(generations),
        desc="[GA gens]",
        file=sys.stdout,
        disable=disable_bar,
        dynamic_ncols=True,
        leave=True,
    )

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/ga_experiment", flush_secs=1)

    # base policy to size params
    base_policy = PolicyNetwork(
        input_dim=int(np.prod(env.observation_space.shape)),
        hidden_dim=64,
        output_dim=env.action_space.n,
        device=device
    )
    base_policy.load_model(model_save_path)
    param_size = len(base_policy.get_params())

    # initialize population
    population = [
        np.array(base_policy.get_params(), dtype=np.float32)
        for _ in range(population_size)
    ]

    # initial fitness logging
    init_fits = [evaluate_fitness(ind, env, device) for ind in population]
    init_min, init_avg, init_med, init_max = (
        float(np.min(init_fits)),
        float(np.mean(init_fits)),
        float(np.median(init_fits)),
        float(np.max(init_fits))
    )
    print(f"[GA DEBUG] Init fitness → min={init_min:.4f}, avg={init_avg:.4f}, "
          f"median={init_med:.4f}, max={init_max:.4f}")
    writer.add_scalar("GA/InitMinFitness", init_min, 0)
    writer.add_scalar("GA/InitAvgFitness", init_avg, 0)
    writer.add_scalar("GA/InitMedianFitness", init_med, 0)
    writer.add_scalar("GA/InitMaxFitness", init_max, 0)

    best_fitness = -float("inf")
    best_params  = None
    stagnation   = 0

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    gpu_count = torch.cuda.device_count() if str(device).startswith("cuda") else 0

    start_time = time.time()

    try:
        for gen in bar:
            # ─── Evaluate ───────────────────────────────────────
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

            # ─── Stats & Logging ─────────────────────────────────
            gen_min    = float(np.min(fitnesses))
            gen_avg    = float(np.mean(fitnesses))
            gen_med    = float(np.median(fitnesses))
            gen_max    = float(np.max(fitnesses))
            param_std  = float(np.std(np.stack(population)))
            elapsed    = time.time() - start_time

            writer.add_scalar("GA/MinFitness",    gen_min, gen)
            writer.add_scalar("GA/AvgFitness",    gen_avg, gen)
            writer.add_scalar("GA/MedianFitness", gen_med, gen)
            writer.add_scalar("GA/MaxFitness",    gen_max, gen)
            writer.add_histogram("GA/FitnessDist", np.array(fitnesses), gen)
            writer.add_scalar("GA/ParamStd",      param_std, gen)
            writer.add_scalar("GA/ElapsedSeconds",elapsed, gen)
            writer.add_scalar("GA/Stagnation",    stagnation, gen)
            writer.flush()

            bar.set_postfix({
                "min":    f"{gen_min:.1f}",
                "avg":    f"{gen_avg:.1f}",
                "med":    f"{gen_med:.1f}",
                "max":    f"{gen_max:.1f}",
                "std":    f"{param_std:.3f}",
                "stag":   f"{stagnation}"
            })
            if disable_bar:
                print(f"[GA] Gen {gen}: min={gen_min:.2f}, avg={gen_avg:.2f}, "
                      f"median={gen_med:.2f}, max={gen_max:.2f}, std={param_std:.4f}, "
                      f"stagnation={stagnation}")

            # ─── Elitism & Checkpoint ───────────────────────────
            if gen_max > best_fitness:
                best_fitness = gen_max
                best_params  = population[int(np.argmax(fitnesses))].copy()
                base_policy.set_params(best_params)
                base_policy.save_model(model_save_path)
                stagnation = 0
            else:
                stagnation += 1

            # ─── Breed Next Generation ──────────────────────────
            elite_n = max(1, int(0.1 * population_size))
            elites  = [population[i] for i in np.argsort(fitnesses)[-elite_n:]]
            new_pop = elites.copy()
            while len(new_pop) < population_size:
                idxs1 = np.random.choice(population_size, tournament_size, replace=False)
                p1     = population[idxs1[np.argmax([fitnesses[i] for i in idxs1])]]
                idxs2 = np.random.choice(population_size, tournament_size, replace=False)
                p2     = population[idxs2[np.argmax([fitnesses[i] for i in idxs2])]]
                alpha  = np.random.rand()
                child  = alpha * p1 + (1 - alpha) * p2

                mask   = np.random.rand(param_size) < mutation_rate
                child[mask] += np.random.randn(int(mask.sum())) * (
                    mutation_scale * (1 + 0.1 * stagnation)
                )
                new_pop.append(child)

            population = new_pop

    finally:
        writer.close()

    # ─── Return the best agent ────────────────────────────────────────────
    final_agent = PolicyNetwork(
        input_dim=int(np.prod(env.observation_space.shape)),
        hidden_dim=64,
        output_dim=env.action_space.n,
        device=device
    )
    final_agent.set_params(best_params)
    return final_agent, best_fitness, None, None
