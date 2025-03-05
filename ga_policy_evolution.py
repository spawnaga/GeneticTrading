import torch
import torch.nn as nn
import numpy as np
from torch.multiprocessing import Pool, cpu_count, set_start_method
import os

# Ensure the correct multiprocessing method for PyTorch
try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Ignore if already set


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
        return self.net(x)

    def act(self, state):
        """
        Return an action (0,1,2) given a state.
        Picks the argmax output from network as discrete action.
        """
        state = state.to(self.device)
        with torch.no_grad():
            q_values = self.forward(state)
        action = torch.argmax(q_values, dim=-1).item()
        return action

    def get_params(self):
        """
        Flatten model parameters into a single NumPy vector.
        """
        return np.concatenate([p.data.cpu().numpy().ravel() for p in self.parameters()])

    def set_params(self, param_vector):
        """
        Load flat param_vector into model parameters.
        Ensures `param_vector` is always a NumPy array.
        """
        param_vector = np.array(param_vector, dtype=np.float32)
        idx = 0
        for p in self.parameters():
            size = p.numel()
            raw = param_vector[idx: idx + size]
            p.data = torch.from_numpy(raw.reshape(p.shape)).float().to(self.device)
            idx += size

    def save_model(self, file_path):
        """
        Save the model parameters to a file.
        """
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        """
        Load the model parameters from a file if it exists.
        """
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, map_location=self.device))
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}. Starting fresh.")


def evaluate_fitness(param_vector, env, device="cpu"):
    """
    Evaluate an individual policy network's fitness.
    Converts param_vector into a PolicyNetwork before running.
    """
    policy_net = PolicyNetwork(env.observation_dim, 64, env.action_space, device=device)
    policy_net.set_params(param_vector)

    total_reward = 0
    obs = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action = policy_net.act(state_tensor)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    return total_reward


# NEW: Parallel evaluation function using GPUs
def parallel_gpu_eval(args):
    param_vector, env, gpu_id = args
    device = f"cuda:{gpu_id}"
    return evaluate_fitness(param_vector, env, device=device)


def run_ga_evolution(env, population_size=30, generations=20, elite_frac=0.2,
                     mutation_rate=0.1, mutation_scale=0.02, num_workers=None,
                     device="cpu", model_save_path="best_policy.pth"):
    """
    Parallelized Genetic Algorithm to evolve policy network weights.
    Now supports GPU-based parallel fitness evaluation and saving/loading of the best model.
    """
    input_dim = env.observation_dim
    output_dim = env.action_space
    hidden_dim = 64

    if num_workers is None:
        num_workers = min(cpu_count(), 32)  # Adjust based on hardware

    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0

    base_policy = PolicyNetwork(input_dim, hidden_dim, output_dim, device=device)
    base_policy.load_model(model_save_path)

    param_size = len(base_policy.get_params())
    population = [np.array(base_policy.get_params(), dtype=np.float32) for _ in range(population_size)]

    best_fitness = -float('inf')
    best_params = None

    for gen in range(generations):
        if gpu_count > 0:
            with Pool(processes=gpu_count) as pool:
                args_list = [(params, env, idx % gpu_count) for idx, params in enumerate(population)]
                fitnesses = pool.map(parallel_gpu_eval, args_list)
        else:
            with Pool(processes=num_workers) as pool:
                fitnesses = pool.starmap(evaluate_fitness, [(params, env, device) for params in population])

        max_fit = np.max(fitnesses)
        avg_fit = np.mean(fitnesses)
        if max_fit > best_fitness:
            best_fitness = max_fit
            best_idx = np.argmax(fitnesses)
            best_params = population[best_idx].copy()
            base_policy.set_params(best_params)
            base_policy.save_model(model_save_path)

        print(f"Gen {gen} | Best fit: {max_fit:.2f}, Avg fit: {avg_fit:.2f}, Overall best: {best_fitness:.2f}")

        elite_count = int(elite_frac * population_size)
        sorted_indices = np.argsort(fitnesses)[-elite_count:]
        elites = np.array([population[i] for i in sorted_indices])

        new_population = elites.tolist()

        while len(new_population) < population_size:
            p1_idx, p2_idx = np.random.choice(len(elites), 2, replace=False)
            p1, p2 = elites[p1_idx], elites[p2_idx]

            cx_point = np.random.randint(0, param_size)
            child_params = np.concatenate([p1[:cx_point], p2[cx_point:]])

            mutate_mask = np.random.rand(param_size) < mutation_rate
            child_params[mutate_mask] += np.random.randn(np.sum(mutate_mask)) * mutation_scale

            new_population.append(child_params)

        population = new_population

    return base_policy, best_fitness
