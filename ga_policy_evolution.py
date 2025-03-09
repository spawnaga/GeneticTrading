import cudf
import torch
import torch.nn as nn
import numpy as np
from torch.multiprocessing import Pool, cpu_count, set_start_method
import os
import torch.distributed as dist

# Set multiprocessing start method to 'spawn' for PyTorch compatibility
try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Ignore if already set


class PolicyNetwork(nn.Module):
    """
    A neural network policy for trading decisions.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, device="cpu"):
        """
        Initialize the policy network.

        Args:
            input_dim (int): Dimension of the input observation.
            hidden_dim (int): Size of hidden layers.
            output_dim (int): Number of possible actions.
            device (str or torch.device): Device to run the model on (e.g., 'cuda:0').
        """
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
        """Forward pass through the network."""
        return self.net(x)

    def act(self, state):
        """
        Select an action based on the state.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            int: Chosen action index.
        """
        state = state.to(self.device)
        with torch.no_grad():
            q_values = self.forward(state)
        return torch.argmax(q_values, dim=-1).item()

    def get_params(self):
        """
        Flatten model parameters into a NumPy array.

        Returns:
            np.ndarray: Flattened parameter vector.
        """
        return np.concatenate([p.data.cpu().numpy().ravel() for p in self.parameters()])

    def set_params(self, param_vector):
        """
        Set model parameters from a flattened vector.

        Args:
            param_vector (np.ndarray): Flattened parameter vector.
        """
        param_vector = np.array(param_vector, dtype=np.float32)
        idx = 0
        for p in self.parameters():
            size = p.numel()
            raw = param_vector[idx: idx + size]  # Extract NumPy array slice
            p.data = torch.from_numpy(raw.reshape(p.shape)).float().to(self.device)  # Convert correctly
            idx += size

    def save_model(self, file_path):
        """
        Save the model state to a file.

        Args:
            file_path (str): Path to save the model.
        """
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        """
        Load the model state from a file if it exists.

        Args:
            file_path (str): Path to the model file.
        """
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, map_location=self.device))
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}. Starting fresh.")


def evaluate_fitness(param_vector, env, device="cpu"):
    """
    Evaluate the fitness of a policy network with given parameters.

    Args:
        param_vector (np.ndarray): Flattened parameter vector.
        env (TradingEnvironment): The trading environment.
        device (str): Device to run evaluation on (e.g., 'cuda:0').

    Returns:
        float: Total reward accumulated.
    """
    # Use env.action_space.n to get the integer number of actions
    policy_net = PolicyNetwork(env.observation_dim, 64, env.action_space.n, device=device)
    policy_net.set_params(param_vector)

    total_reward = 0.0
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action = policy_net.act(state_tensor)
        obs, reward, done, _ = env.step(action)

        # Handle reward type conversion
        if isinstance(reward, cudf.Series):
            reward = reward.iloc[0].item() if not reward.isnull().any() else 0.0
        elif isinstance(reward, np.ndarray):
            reward = reward.item()
        elif isinstance(reward, (int, float, np.float32, np.float64)):
            reward = float(reward)
        else:
            reward = float(reward[0])  # Fallback
        total_reward += reward
        steps += 1
        # print(f"observation_dim: {env.observation_dim}, action_space.n: {env.action_space.n}")
        # print(f"Evaluating on {device}, step {steps}, current reward: {reward:.5f}, total reward: {total_reward:.4f}")

    print(f"Evaluation completed on {device} - Total Reward: {total_reward:.3f}, Steps: {steps}")
    return total_reward


def parallel_gpu_eval(args):
    """
    Wrapper for parallel GPU evaluation.

    Args:
        args (tuple): (param_vector, env, gpu_id)

    Returns:
        float: Fitness score.
    """
    param_vector, env, gpu_id = args
    device = f"cuda:{gpu_id}"
    return evaluate_fitness(param_vector, env, device=device)


def run_ga_evolution(env, population_size=30, generations=20, elite_frac=0.2,
                     mutation_rate=0.1, mutation_scale=0.02, num_workers=None,
                     device="cpu", model_save_path="best_policy.pth", distributed=False, local_rank=0, world_size=1):
    """
    Run a Genetic Algorithm to evolve a policy network.

    Args:
        env (TradingEnvironment): Trading environment instance.
        population_size (int): Number of individuals in the population.
        generations (int): Number of generations to evolve.
        elite_frac (float): Fraction of top individuals to keep as elites.
        mutation_rate (float): Probability of mutating a parameter.
        mutation_scale (float): Scale of mutation noise.
        num_workers (int, optional): Number of worker processes for parallel evaluation.
        device (str): Device for model operations (e.g., 'cuda:0').
        model_save_path (str): Path to save the best model.
        distributed (bool): Whether to use distributed evaluation across multiple GPUs.
        local_rank (int): Local rank of the process in distributed training.
        world_size (int): Total number of processes in distributed training.

    Returns:
        tuple: (best_agent, best_fitness)
    """
    input_dim = env.observation_dim
    output_dim = env.action_space.n
    hidden_dim = 64

    # Set number of workers if not specified
    if num_workers is None:
        num_workers = min(cpu_count(), 32)

    gpu_count = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    base_policy = PolicyNetwork(input_dim, hidden_dim, output_dim, device=device)
    base_policy.load_model(model_save_path)

    param_size = len(base_policy.get_params())
    population = [
        np.array(PolicyNetwork(input_dim, hidden_dim, output_dim, device=device).get_params(), dtype=np.float32)
        for _ in range(population_size)]

    best_fitness = -float('inf')
    best_params = None

    for gen in range(generations):
        if distributed:
            # Split population across ranks
            chunk_size = population_size // world_size
            start_idx = local_rank * chunk_size
            end_idx = start_idx + chunk_size if local_rank != world_size - 1 else population_size
            local_population = population[start_idx:end_idx]
        else:
            local_population = population

        # Parallel evaluation with GPUs if available
        if gpu_count > 0:
            with Pool(processes=gpu_count) as pool:
                args_list = [(params, env, idx % gpu_count) for idx, params in enumerate(local_population)]
                fitnesses = pool.map(parallel_gpu_eval, args_list)
        else:
            with Pool(processes=num_workers) as pool:
                fitnesses = pool.starmap(evaluate_fitness, [(params, env, device) for params in local_population])

        if distributed:
            # Gather fitness scores to rank 0
            all_fitnesses = [None] * population_size if local_rank == 0 else None
            local_fitness_tensor = torch.tensor(fitnesses, dtype=torch.float32).to(device)
            gather_list = [torch.zeros(chunk_size, dtype=torch.float32).to(device) for _ in range(world_size)] if local_rank == 0 else None
            dist.gather(local_fitness_tensor, gather_list=gather_list if local_rank == 0 else None, dst=0)
            if local_rank == 0:
                fitnesses = torch.cat(gather_list[:world_size-1] + [torch.zeros(population_size - (world_size-1) * chunk_size).to(device)]).tolist()
            dist.barrier()

        if local_rank == 0 or not distributed:
            # Update best individual
            max_fit = np.max(fitnesses)
            avg_fit = np.mean(fitnesses)
            if max_fit > best_fitness:
                best_fitness = max_fit
                best_idx = np.argmax(fitnesses)
                best_params = population[best_idx].copy()
                base_policy.set_params(best_params)
                base_policy.save_model(model_save_path)

            print(f"Gen {gen} | Best fit: {max_fit:.2f}, Avg fit: {avg_fit:.2f}, Overall best: {best_fitness:.2f}")

            # Elitism: Keep top performers
            elite_count = int(elite_frac * population_size)
            sorted_indices = np.argsort(fitnesses)[-elite_count:]
            elites = np.array([population[i] for i in sorted_indices])

            # Generate new population
            new_population = elites.tolist()
            while len(new_population) < population_size:
                p1_idx, p2_idx = np.random.choice(len(elites), 2, replace=False)
                p1, p2 = elites[p1_idx], elites[p2_idx]

                # Crossover
                cx_point = np.random.randint(0, param_size)
                child_params = np.concatenate([p1[:cx_point], p2[cx_point:]])

                # Mutation
                mutate_mask = np.random.rand(param_size) < mutation_rate
                child_params[mutate_mask] += np.random.randn(np.sum(mutate_mask)) * mutation_scale

                new_population.append(child_params)

            population = new_population

        if distributed:
            # Broadcast updated population to all ranks
            for i, params in enumerate(population):
                params_tensor = torch.tensor(params, dtype=torch.float32).to(device)
                dist.broadcast(params_tensor, src=0)
                population[i] = params_tensor.cpu().numpy()

    # Return the best agent
    if local_rank == 0:
        best_agent = PolicyNetwork(input_dim, hidden_dim, output_dim, device=device)
        best_agent.set_params(best_params)
        return best_agent, best_fitness
    return None, best_fitness # Non-rank 0 returns None for the agent
