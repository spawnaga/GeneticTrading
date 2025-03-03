import torch
import torch.nn as nn
import numpy as np
from torch.multiprocessing import Pool, cpu_count, set_start_method

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
        param_vector = np.array(param_vector, dtype=np.float32)  # ✅ Ensure it's a NumPy array
        idx = 0
        for p in self.parameters():
            size = p.numel()
            raw = param_vector[idx: idx + size]  # ✅ Extract NumPy array slice
            p.data = torch.from_numpy(raw.reshape(p.shape)).float().to(self.device)  # ✅ Convert correctly
            idx += size

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

def run_ga_evolution(env, population_size=30, generations=20, elite_frac=0.2, mutation_rate=0.1, mutation_scale=0.02, num_workers=None, device="cpu"):
    """
    Parallelized Genetic Algorithm to evolve policy network weights.
    Uses `torch.multiprocessing` to enable multi-GPU execution.
    """
    input_dim = env.observation_dim
    output_dim = env.action_space
    hidden_dim = 64

    # Use all available CPU threads if num_workers is not specified
    if num_workers is None:
        num_workers = min(cpu_count(), 32)  # Adjust based on your hardware

    # Initialize population as parameter vectors
    base_policy = PolicyNetwork(input_dim, hidden_dim, output_dim, device=device)
    param_size = len(base_policy.get_params())
    population = [np.array(PolicyNetwork(input_dim, hidden_dim, output_dim, device=device).get_params(), dtype=np.float32) for _ in range(population_size)]

    best_fitness = -float('inf')
    best_params = None

    for gen in range(generations):
        # ✅ Parallel fitness evaluation with multiprocessing
        with Pool(processes=num_workers) as pool:
            fitnesses = pool.starmap(evaluate_fitness, [(params, env, device) for params in population])

        # Track best individual
        max_fit = np.max(fitnesses)
        avg_fit = np.mean(fitnesses)
        if max_fit > best_fitness:
            best_fitness = max_fit
            best_idx = np.argmax(fitnesses)
            best_params = population[best_idx].copy()

        print(f"Gen {gen} | Best fit: {max_fit:.2f}, Avg fit: {avg_fit:.2f}, Overall best: {best_fitness:.2f}")

        # Selection: Keep the best elite_frac% individuals
        elite_count = int(elite_frac * population_size)
        sorted_indices = np.argsort(fitnesses)[-elite_count:]
        elites = [population[i] for i in sorted_indices]

        # ✅ Convert elites to a NumPy array to ensure proper selection
        elites = np.array(elites)  # Shape: (elite_count, param_size)

        # Create new population
        new_population = elites.tolist()  # ✅ Convert back to list

        while len(new_population) < population_size:
            # ✅ Fix the selection of parents
            p1_idx, p2_idx = np.random.choice(len(elites), 2, replace=False)
            p1, p2 = elites[p1_idx], elites[p2_idx]

            # Single-point crossover
            cx_point = np.random.randint(0, param_size)
            child_params = np.concatenate([p1[:cx_point], p2[cx_point:]])

            # Mutation
            mutate_mask = np.random.rand(param_size) < mutation_rate
            child_params[mutate_mask] += np.random.randn(np.sum(mutate_mask)) * mutation_scale

            new_population.append(child_params)

        population = new_population

    # ✅ Convert best params back into a PolicyNetwork
    best_agent = PolicyNetwork(input_dim, hidden_dim, output_dim, device=device)
    best_agent.set_params(best_params)
    return best_agent, best_fitness
