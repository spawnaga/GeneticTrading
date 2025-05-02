# ga_policy_evolution.py

import os
import gym
import torch
import torch.nn as nn
import numpy as np
from torch.multiprocessing import Pool, cpu_count, set_start_method
from torch.utils.tensorboard import SummaryWriter

# Use 'spawn' to avoid CUDA issues when forking
try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass


class PolicyNetwork(nn.Module):
    """
    A simple 3‐layer MLP policy network for GA evolution.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, device="cpu"):
        super().__init__()
        # store as torch.device for consistency
        self.device = torch.device(device)
        # build the network on the correct device
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        ).to(self.device)

    def forward(self, x):
        # Flatten if extra dims
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

    def act(self, state):
        """
        Pick the action with the highest logit.
        """
        state = state.to(self.device)
        with torch.no_grad():
            logits = self.forward(state)
        return int(torch.argmax(logits, dim=-1).item())

    def get_params(self):
        """
        Flatten and return all parameters as a single numpy vector.
        """
        return np.concatenate([p.cpu().data.numpy().ravel() for p in self.parameters()])

    def set_params(self, vector):
        """
        Load flattened numpy vector back into model parameters.
        """
        vec = np.array(vector, dtype=np.float32)
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            chunk = vec[idx:idx + numel].reshape(p.shape)
            p.data.copy_(torch.from_numpy(chunk).to(self.device))
            idx += numel

    def save_model(self, path):
        """Checkpoint to disk."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load checkpoint if it exists."""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=self.device))
            print(f"[GA] Loaded existing model from {path}")
        else:
            print(f"[GA] No model found at {path}, starting fresh.")


def evaluate_fitness(param_vector, env, device="cpu"):
    """
    Roll out one episode with the given parameters, returning total (scaled) reward.
    """
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("GA policy only supports Box observation spaces")

    # Use the env’s observation_space to get correct input_dim
    input_dim = int(np.prod(env.observation_space.shape))
    policy = PolicyNetwork(input_dim, 64, env.action_space.n, device=device)
    policy.set_params(param_vector)

    total_reward = 0.0
    obs = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy.device)
        action = policy.act(state_tensor)
        obs, reward, done, _ = env.step(action)
        # scale to accentuate differences
        total_reward += float(reward) * 10.0

    return total_reward


def parallel_gpu_eval(args):
    """
    Helper to run evaluate_fitness on a specific GPU.
    """
    params, env, gpu_id = args
    return evaluate_fitness(params, env, device=f"cuda:{gpu_id}")


def run_ga_evolution(
    env,
    population_size=40,
    generations=20,
    tournament_size=3,
    mutation_rate=0.5,
    mutation_scale=0.2,
    num_workers=None,
    device="cpu",
    model_save_path="best_ga_policy.pth"
):
    """
    Genetic‐Algorithm main loop with:
      - elitism (top 10%)
      - tournament selection
      - weighted crossover
      - adaptive mutation
      - TensorBoard logging & detailed debug scalars/histograms
    """

    # verify the env
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("GA policy only supports Box observation spaces")

    input_dim = int(np.prod(env.observation_space.shape))
    output_dim = env.action_space.n

    # set up TensorBoard writer; flush every 5 seconds
    writer = SummaryWriter(log_dir="runs/ga_experiment", flush_secs=5)

    # load or initialize the “base” policy for checkpointing
    base_policy = PolicyNetwork(input_dim, 64, output_dim, device=device)
    base_policy.load_model(model_save_path)
    param_size = len(base_policy.get_params())

    # initialize a random population
    population = [
        np.array(PolicyNetwork(input_dim, 64, output_dim, device=device)
                 .get_params(), dtype=np.float32)
        for _ in range(population_size)
    ]

    # --- initial fitness debug ---
    init_fits = [evaluate_fitness(ind, env, device) for ind in population]
    print(f"[GA DEBUG] Init fitness → "
          f"min={min(init_fits):.4f}, "
          f"avg={np.mean(init_fits):.4f}, "
          f"max={max(init_fits):.4f}")
    writer.add_scalar("GA/InitMinFitness", min(init_fits), 0)
    writer.add_scalar("GA/InitAvgFitness", np.mean(init_fits), 0)
    writer.add_scalar("GA/InitMaxFitness", max(init_fits), 0)

    best_fitness = -float("inf")
    best_params = None
    stagnation = 0

    # set up parallelism
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    device_str = str(device)
    gpu_count = torch.cuda.device_count() if device_str.startswith("cuda") else 0

    try:
        for gen in range(generations):
            # --- evaluate population ---
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

            gen_best = float(np.max(fitnesses))
            gen_avg  = float(np.mean(fitnesses))

            # log scalars
            writer.add_scalar("GA/BestFitness", gen_best, gen)
            writer.add_scalar("GA/AvgFitness",  gen_avg,  gen)
            # log histogram of fitness
            writer.add_histogram("GA/FitnessDist", np.array(fitnesses), gen)

            # track parameter diversity
            pop_matrix = np.stack(population)
            param_std = float(np.std(pop_matrix))
            writer.add_scalar("GA/ParamStd", param_std, gen)

            # forced flush & debug print
            writer.flush()
            print(f"[GA] Gen {gen}: "
                  f"best={gen_best:.2f}, avg={gen_avg:.2f}, "
                  f"all_time_best={best_fitness:.2f}, param_std={param_std:.4f}")

            # --- elitism & checkpointing ---
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_idx     = int(np.argmax(fitnesses))
                best_params  = population[best_idx].copy()
                base_policy.set_params(best_params)
                base_policy.save_model(model_save_path)
                stagnation = 0
            else:
                stagnation += 1

            # --- create next generation ---
            elite_n = max(1, int(0.1 * population_size))
            elites  = [population[i] for i in np.argsort(fitnesses)[-elite_n:]]

            new_pop = elites.copy()
            while len(new_pop) < population_size:
                # tournament selection for two parents
                idxs1 = np.random.choice(population_size, tournament_size, replace=False)
                p1 = population[idxs1[np.argmax([fitnesses[i] for i in idxs1])]]
                idxs2 = np.random.choice(population_size, tournament_size, replace=False)
                p2 = population[idxs2[np.argmax([fitnesses[i] for i in idxs2])]]
                alpha = np.random.rand()
                child = alpha * p1 + (1 - alpha) * p2

                # adaptive mutation
                scale = mutation_scale * (1 + 0.1 * stagnation)
                mask  = np.random.rand(param_size) < mutation_rate
                child[mask] += np.random.randn(int(mask.sum())) * scale

                new_pop.append(child)

            population = new_pop

    finally:
        writer.close()

    # return the best‐found policy plus stats
    final_agent = PolicyNetwork(input_dim, 64, output_dim, device=device)
    final_agent.set_params(best_params)
    return final_agent, best_fitness, None, None  # histories no longer returned
