import gym
import torch
import torch.nn as nn
import numpy as np
from torch.multiprocessing import Pool, cpu_count, set_start_method
import os
import torch.distributed as dist

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device="cpu"):
        super(PolicyNetwork, self).__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        ).to(self.device)

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

    def act(self, state):
        state = state.to(self.device)
        with torch.no_grad():
            q_values = self.forward(state)
        return torch.argmax(q_values, dim=-1).item()

    def get_params(self):
        return np.concatenate([p.data.cpu().numpy().ravel() for p in self.parameters()])

    def set_params(self, param_vector):
        param_vector = np.array(param_vector, dtype=np.float32)
        idx = 0
        for p in self.parameters():
            size = p.numel()
            raw = param_vector[idx:idx + size]
            p.data = torch.from_numpy(raw.reshape(p.shape)).float().to(self.device)
            idx += size

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, map_location=self.device))
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}. Starting fresh.")


def evaluate_fitness(param_vector, env, device="cpu"):
    if isinstance(env.observation_space, gym.spaces.Box):
        input_dim = int(np.prod(env.observation_space.shape))
    else:
        raise ValueError("Observation space must be a Box for this GA code.")

    policy_net = PolicyNetwork(input_dim, 64, env.action_space.n, device=device)
    policy_net.set_params(param_vector)

    total_reward = 0.0
    obs = env.reset()
    done = False

    while not done:
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action = policy_net.act(state_tensor)
        obs, reward, done, _ = env.step(action)
        # Scale reward if you want to emphasize differences
        total_reward += float(reward) * 10

    return total_reward

def parallel_gpu_eval(args):
    param_vector, env, gpu_id = args
    device = f"cuda:{gpu_id}"
    return evaluate_fitness(param_vector, env, device=device)

def run_ga_evolution(env, population_size=40, generations=20, elite_frac=0.2,
                     mutation_rate=0.5, mutation_scale=0.2, num_workers=None,
                     device="cpu", model_save_path="best_policy.pth",
                     distributed=False, local_rank=0, world_size=1):

    if isinstance(env.observation_space, gym.spaces.Box):
        input_dim = int(np.prod(env.observation_space.shape))
    else:
        raise ValueError("Observation space must be a Box for this GA code.")
    output_dim = env.action_space.n
    hidden_dim = 64

    if num_workers is None:
        num_workers = min(cpu_count(), 32)

    base_policy = PolicyNetwork(input_dim, hidden_dim, output_dim, device=device)
    base_policy.load_model(model_save_path)

    param_size = len(base_policy.get_params())
    population = [
        np.array(PolicyNetwork(input_dim, hidden_dim, output_dim, device=device).get_params(), dtype=np.float32)
        for _ in range(population_size)
    ]

    # Sync initial population if distributed
    if distributed:
        for i, params in enumerate(population):
            params_tensor = torch.tensor(params, dtype=torch.float32).to(device)
            dist.broadcast(params_tensor, src=0)
            population[i] = params_tensor.cpu().numpy()

    best_fitness = -float('inf')
    best_params = None
    stagnation_count = 0

    for gen in range(generations):
        # Shard population if distributed
        if distributed:
            chunk_size = population_size // world_size
            remainder = population_size % world_size
            if local_rank < remainder:
                chunk_size += 1
                start_idx = local_rank * chunk_size
                end_idx = start_idx + chunk_size
            else:
                start_idx = local_rank * chunk_size + remainder
                end_idx = start_idx + chunk_size
            local_population = population[start_idx:end_idx]
        else:
            local_population = population

        gpu_count = torch.cuda.device_count() if str(device).startswith("cuda") else 0

        if gpu_count > 0:
            with Pool(processes=gpu_count) as pool:
                args_list = [(params, env, idx % gpu_count) for idx, params in enumerate(local_population)]
                local_fitnesses = pool.map(parallel_gpu_eval, args_list)
        else:
            if num_workers > 1:
                with Pool(processes=num_workers) as pool:
                    local_fitnesses = pool.starmap(evaluate_fitness, [(params, env, device) for params in local_population])
            else:
                local_fitnesses = [evaluate_fitness(params, env, device) for params in local_population]

        # Gather fitness from all ranks
        if distributed:
            local_fitness_tensor = torch.tensor(local_fitnesses, dtype=torch.float32).to(device)
            gather_list = []
            if local_rank == 0:
                for rank in range(world_size):
                    rank_chunk_size = population_size // world_size + (1 if rank < population_size % world_size else 0)
                    gather_list.append(torch.zeros(rank_chunk_size, dtype=torch.float32).to(device))
            dist.gather(local_fitness_tensor, gather_list=gather_list if local_rank == 0 else None, dst=0)

            if local_rank == 0:
                fitnesses = [f.item() for sublist in gather_list for f in sublist]
            else:
                fitnesses = None
            dist.barrier()
        else:
            fitnesses = local_fitnesses

        if (local_rank == 0) or (not distributed):
            max_fit = np.max(fitnesses)
            avg_fit = np.mean(fitnesses)
            if max_fit > best_fitness:
                best_fitness = max_fit
                best_idx = np.argmax(fitnesses)
                best_params = population[best_idx].copy()
                base_policy.set_params(best_params)
                base_policy.save_model(model_save_path)
                stagnation_count = 0
            else:
                stagnation_count += 1

            print(f"Gen {gen} | Best fit: {max_fit:.2f}, Avg fit: {avg_fit:.2f}, Overall best: {best_fitness:.2f}")
            top_fitnesses = sorted(fitnesses, reverse=True)[:5]
            print(f"Top 5 fitnesses: {top_fitnesses}")

            elite_count = int(elite_frac * population_size)
            sorted_indices = np.argsort(fitnesses)[-elite_count:]
            elites = np.array([population[i] for i in sorted_indices])

            new_population = elites.tolist()
            while len(new_population) < population_size:
                p1_idx, p2_idx = np.random.choice(len(elites), 2, replace=False)
                p1, p2 = elites[p1_idx], elites[p2_idx]
                mask = np.random.rand(param_size) < 0.5
                child_params = np.where(mask, p1, p2)
                current_mutation_scale = mutation_scale * (1 + stagnation_count * 0.1)
                mutate_mask = np.random.rand(param_size) < mutation_rate
                child_params[mutate_mask] += np.random.randn(np.sum(mutate_mask)) * current_mutation_scale
                new_population.append(child_params)

            population = new_population

        # Broadcast updated population
        if distributed:
            for i, params in enumerate(population):
                params_tensor = torch.tensor(params, dtype=torch.float32).to(device)
                dist.broadcast(params_tensor, src=0)
                population[i] = params_tensor.cpu().numpy()
            dist.barrier()

    if local_rank == 0:
        best_agent = PolicyNetwork(input_dim, hidden_dim, output_dim, device=device)
        best_agent.set_params(best_params)
        return best_agent, best_fitness
    return None, best_fitness