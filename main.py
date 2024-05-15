import os
import numpy as np
import pandas as pd
from deap import creator, base, tools, algorithms
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import random_tf_policy
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import tf_py_environment
from skopt import gp_minimize, space
from tqdm import tqdm

# Define a custom trading environment
class CustomTradingEnvironment(py_environment.PyEnvironment):
    def __init__(self, data):
        super().__init__()
        self.data = data.astype(np.float32)
        self._index = 0
        self._episode_ended = False
        self.position = 0  # 0: no position, 1: long, -1: short

        self._observation_spec = array_spec.BoundedArraySpec(shape=(6,), dtype=np.float32, minimum=-1, maximum=1)
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2)
        self._state = self._get_state()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._index = 0
        self._episode_ended = False
        self.position = 0  # Reset position
        self._state = self._get_state()
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._index += 1
        self._state = self._get_state()

        price_change = self.data.iloc[self._index]['lastPrice'] - self.data.iloc[self._index - 1]['lastPrice']
        reward = 0

        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
            elif self.position == -1:
                reward += price_change  # Profit from short
                self.position = 0
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
            elif self.position == 1:
                reward -= price_change  # Profit from long
                self.position = 0
        else:  # Hold
            if self.position == 1:
                reward += price_change  # Profit from long
            elif self.position == -1:
                reward -= price_change  # Profit from short

        if self._index >= len(self.data) - 1:
            self._episode_ended = True

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)

    def _get_state(self):
        if self._index < len(self.data):
            return self.data.iloc[self._index].values.astype(np.float32)
        else:
            return np.zeros(6, dtype=np.float32)

# Step 1: Data fetching and chunk processing
def fetch_data_in_chunks(database_url, query, chunksize=10000):
    engine = create_engine(database_url)
    with engine.connect() as connection:
        chunk_iterator = pd.read_sql_query(query, connection, chunksize=chunksize)
        for chunk in chunk_iterator:
            yield chunk

# Evaluate the model
def evaluate_model(agent, env, num_episodes=10):
    returns = []
    for _ in range(num_episodes):
        time_step = env.reset()
        episode_return = 0
        while not time_step.is_last():
            action_step = agent.agent.policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward
        returns.append(episode_return.numpy())
    avg_return = np.mean(returns)
    return avg_return

def load_and_process_data_in_chunks(database_path, query, chunksize=10000):
    database_url = f"sqlite:///{database_path}"
    processed_chunks = []

    def process_chunk(chunk):
        processor = MarketDataProcessor(chunk)
        return processor.preprocess()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for chunk in tqdm(fetch_data_in_chunks(database_url, query, chunksize),
                          desc="Loading and Processing Data Chunks"):
            futures.append(executor.submit(process_chunk, chunk))

        for future in tqdm(futures, desc="Combining Processed Chunks"):
            processed_chunks.append(future.result())

    # Combine processed chunks into a single DataFrame
    return pd.concat(processed_chunks, ignore_index=True)

# Step 2: Data preprocessing class
class MarketDataProcessor:
    def __init__(self, data):
        self.data = data.copy()  # Create a copy to avoid SettingWithCopyWarning

    def preprocess(self):
        # Handle datetime parsing errors by using errors='coerce' to convert invalid parsing to NaT
        self.data['time'] = pd.to_datetime(self.data['time'], errors='coerce')

        # Check for any rows with NaT values and handle them (e.g., drop or fill)
        self.data = self.data.dropna(subset=['time'])  # Drop rows with NaT values in the 'time' column

        # Use .loc to avoid SettingWithCopyWarning
        self.data['time_of_day'] = self.data['time'].dt.hour + self.data['time'].dt.minute / 60
        self.data['spread'] = self.data['askPrice_1'] - self.data['bidPrice_1']
        self.data['order_imbalance'] = (self.data['askSize_1'] - self.data['bidSize_1']) / (
                    self.data['askSize_1'] + self.data['bidSize_1'])
        self.data['day_of_week'] = self.data['time'].dt.dayofweek

        # Clip all columns to handle extreme values
        self.data = self.clip_extreme_values(self.data)

        # Remove rows with any NaN values
        self.data = self.data.dropna()

        # Scaling the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(
            self.data[['lastPrice', 'lastSize', 'spread', 'order_imbalance', 'time_of_day', 'day_of_week']].astype(np.float32)
        )
        self.data[['lastPrice', 'lastSize', 'spread', 'order_imbalance', 'time_of_day', 'day_of_week']] = scaled_features

        return self.data[['lastPrice', 'lastSize', 'spread', 'order_imbalance', 'time_of_day', 'day_of_week']].astype(np.float32)

    def summarize_spread(self):
        print("Summary statistics for 'spread':")
        print(self.data['spread'].describe())

    def clip_extreme_values(self, data):
        # Clip extreme values in all relevant columns to a reasonable range
        data['lastPrice'] = data['lastPrice'].clip(lower=data['lastPrice'].quantile(0.01),
                                                   upper=data['lastPrice'].quantile(0.99))
        data['lastSize'] = data['lastSize'].clip(lower=data['lastSize'].quantile(0.01),
                                                 upper=data['lastSize'].quantile(0.99))
        data['spread'] = data['spread'].clip(lower=data['spread'].quantile(0.01), upper=data['spread'].quantile(0.99))
        data['order_imbalance'] = data['order_imbalance'].clip(lower=data['order_imbalance'].quantile(0.01),
                                                               upper=data['order_imbalance'].quantile(0.99))
        data['time_of_day'] = data['time_of_day'].clip(lower=data['time_of_day'].quantile(0.01),
                                                       upper=data['time_of_day'].quantile(0.99))
        data['day_of_week'] = data['day_of_week'].clip(lower=data['day_of_week'].quantile(0.01),
                                                       upper=data['day_of_week'].quantile(0.99))
        return data

# Step 3: Bayesian Optimization for Hyperparameter Tuning
class BayesianOptimization:
    def __init__(self, search_space, objective_function):
        self.search_space = search_space
        self.objective_function = objective_function

    def optimize(self, n_calls=50):
        res = gp_minimize(self.objective_function, self.search_space, n_calls=n_calls, random_state=42)
        return res

# Reinforcement Learning class
class ReinforcementLearning:
    def __init__(self, environment, learning_rate, discount_factor):
        self.environment = environment
        self.train_env = tf_py_environment.TFPyEnvironment(environment)
        self.eval_env = tf_py_environment.TFPyEnvironment(environment)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.agent = self.setup_agent()
        self.replay_buffer = self.setup_replay_buffer()
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=64,
            num_steps=2
        ).prefetch(3)
        self.iterator = iter(self.dataset)

    def setup_agent(self):
        q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=(100,)
        )

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step_counter = tf.compat.v2.Variable(0)

        agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,
            gamma=self.discount_factor
        )
        agent.initialize()
        return agent

    def setup_replay_buffer(self):
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=100000
        )
        return replay_buffer

    def collect_data(self, policy, steps):
        driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
            policy,
            observers=[self.replay_buffer.add_batch],
            num_steps=steps
        )
        final_time_step, _ = driver.run()
        return final_time_step

    def train_agent(self, num_iterations=10000, collect_steps_per_iteration=1):
        random_policy = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(),
            self.train_env.action_spec()
        )

        # Collect initial data
        for _ in tqdm(range(1000), desc="Collecting Initial Data"):
            self.collect_data(random_policy, collect_steps_per_iteration)

        # Training loop
        for _ in tqdm(range(num_iterations), desc="Training Agent"):
            self.collect_data(self.agent.collect_policy, collect_steps_per_iteration)
            experience, unused_info = next(self.iterator)
            train_loss = self.agent.train(experience)
            step = self.agent.train_step_counter.numpy()

            if step % 100 == 0:
                print(f'Step = {step}: Loss = {train_loss.loss.numpy()}')

    def save_agent(self, checkpoint_dir):
        checkpoint = tf.train.Checkpoint(agent=self.agent)
        checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "checkpoint"))

    def load_agent(self, checkpoint_dir):
        checkpoint = tf.train.Checkpoint(agent=self.agent)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            print("Agent restored from checkpoint:", latest_checkpoint)
        else:
            print("No checkpoint found. Initializing agent from scratch.")

# Step 5: Genetic Algorithm for Further Optimization
class GeneticAlgorithm:
    def __init__(self):
        self.toolbox = base.Toolbox()
        self.setup_toolbox()

    def setup_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox.register("attribute", np.random.random)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attribute, n=10)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(self, individual):
        # Placeholder for actual evaluation logic
        return sum(individual),

    def run_evolution(self):
        population = self.toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        algorithms.eaSimple(population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof,
                            verbose=True)
        return hof[0]

# Step 6: Objective Function for Bayesian Optimization
def objective(params):
    learning_rate, discount_factor = params
    # Initialize RL environment within the function scope
    processed_data = load_and_process_data_in_chunks(database_path, query, chunksize)
    rl_env = CustomTradingEnvironment(processed_data)
    rl_agent = ReinforcementLearning(rl_env, learning_rate, discount_factor)
    rl_agent.train_agent(num_iterations=1000, collect_steps_per_iteration=1)
    return -rl_agent.agent.train_step_counter.numpy()  # Negative because we want to maximize the steps

# Step 7: Main function to integrate everything
def main():
    global database_path, query, chunksize
    database_path = './ES_ticks.db'
    query = "SELECT * FROM ES_market_depth"
    chunksize = 10000  # Adjust the chunk size based on your memory constraints

    processed_data = load_and_process_data_in_chunks(database_path, query, chunksize)
    print("Loaded and processed data shape:", processed_data.shape)  # Debugging: Print the shape of the processed data

    # Define the search space for Bayesian Optimization
    search_space = [space.Real(1e-5, 1e-1, prior='log-uniform', name='learning_rate'),
                    space.Real(0.9, 0.999, name='discount_factor')]

    # Bayesian Optimization
    bayes_opt = BayesianOptimization(search_space, objective)
    res = bayes_opt.optimize(n_calls=10)
    best_params = res.x
    print(f"Best parameters from Bayesian Optimization: {best_params}")

    # Train RL agent with optimized parameters
    learning_rate, discount_factor = best_params
    rl_env = CustomTradingEnvironment(processed_data)
    rl_agent = ReinforcementLearning(rl_env, learning_rate, discount_factor)
    rl_agent.train_agent(num_iterations=1000, collect_steps_per_iteration=1)

    # Genetic Algorithm Optimization
    ga_opt = GeneticAlgorithm()
    best_individual = ga_opt.run_evolution()
    print(f"Best individual from Genetic Algorithm: {best_individual}")

    tf_env = tf_py_environment.TFPyEnvironment(rl_env)

    rl_agent = ReinforcementLearning(rl_env, learning_rate, discount_factor)

    avg_return = evaluate_model(rl_agent.agent, tf_env, num_episodes=1)
    print(f"Average return over 10 episodes: {avg_return}")

if __name__ == "__main__":
    main()
