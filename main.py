import random
import numpy as np
import pandas as pd
from dateutil import parser
from sklearn.preprocessing import StandardScaler
from deap import creator, base, tools, algorithms
from concurrent.futures import ThreadPoolExecutor
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from sqlalchemy import create_engine
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common


# Define market data processor
class MarketDataProcessor:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        with ThreadPoolExecutor(max_workers=10) as executor:
            self.data['time'] = list(executor.map(parser.parse, self.data['time']))
        self.data['time_of_day'] = self.data['time'].dt.hour + self.data['time'].dt.minute / 60
        self.data['spread'] = self.data['askPrice_1'] - self.data['bidPrice_1']
        self.data['order_imbalance'] = (self.data['askSize_1'] - self.data['bidSize_1']) / (
                self.data['askSize_1'] + self.data['bidSize_1'])
        self.data['day_of_week'] = self.data['time'].dt.dayofweek

        # Scaling the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(
            self.data[['lastPrice', 'lastSize', 'spread', 'order_imbalance', 'time_of_day', 'day_of_week']])
        self.data[
            ['lastPrice', 'lastSize', 'spread', 'order_imbalance', 'time_of_day', 'day_of_week']] = scaled_features

        return self.data[['lastPrice', 'lastSize', 'spread', 'order_imbalance', 'time_of_day', 'day_of_week']]


# Define trading environment
class TradingEnvironment(py_environment.PyEnvironment):
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.n_steps = len(data)
        self.current_step = 0
        self._action_space = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=2,
            name='action'
        )
        self._observation_space = array_spec.BoundedArraySpec(
            shape=(6,),
            dtype=np.float32,
            minimum=-float('inf'),
            maximum=float('inf'),
            name='observation'
        )

    def action_spec(self):
        return self._action_space

    def observation_spec(self):
        return self._observation_space

    def _reset(self):
        self.current_step = 0
        return ts.restart(self.data.iloc[0].values)

    def _step(self, action):
        if self.current_step >= self.n_steps - 1:
            return ts.termination(self.data.iloc[self.current_step].values,
                                  self.calculate_reward(action, self.current_step))
        self.current_step += 1
        reward = self.calculate_reward(action, self.current_step)
        obs = self.data.iloc[self.current_step].values
        return ts.transition(obs, reward=reward)

    def calculate_reward(self, action, step):
        price_change = self.data.iloc[step]['lastPrice']
        if action == 1:
            return price_change
        elif action == 2:
            return -price_change
        return -0.01


# Define DQN agent setup
class DQNAgentSetup:
    def __init__(self, environment):
        self.environment = environment
        self.agent = self.initialize_agent()

    def initialize_agent(self):
        q_net = q_network.QNetwork(
            self.environment.observation_spec(),
            self.environment.action_spec(),
            fc_layer_params=(100, 50)
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        train_step_counter = tf.Variable(0)
        agent = dqn_agent.DqnAgent(
            self.environment.time_step_spec(),
            self.environment.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter
        )
        agent.initialize()
        return agent


@use_named_args(dimensions=[Real(0.0001, 0.1, name='learning_rate'), Real(10, 100, name='num_neurons')])
def evaluate_agent(learning_rate, num_neurons):
    # Assuming processed_data is available in this scope
    env = TradingEnvironment(processed_data)
    agent_setup = DQNAgentSetup(env)
    # Modify the agent's network or parameters based on params
    return np.random.random()  # Placeholder for actual evaluation


# Assume a simple fitness strategy and individual structure for demonstration
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator: defining how to create a random float attribute
toolbox.register("attr_float", random.uniform, 0.0, 1.0)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float,
                 n=10)  # n=10 attributes per individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_agent)  # Define your evaluation function

# Read data and preprocess
database_path = '../market_direction_prediction/ES_ticks.db'
engine = create_engine(f"sqlite:///{database_path}")
data = pd.read_sql("SELECT * from ES_market_depth", engine)
processor = MarketDataProcessor(data)
processed_data = processor.preprocess()


# Now you can call run_genetic_algorithm without getting the AttributeError
def run_genetic_algorithm():
    population = toolbox.population(n=100)  # Create a population of 100 individuals
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    result = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof,
                                 verbose=True)
    best_ind = hof.items[0]
    print(f'Best individual: {best_ind}, Fitness: {best_ind.fitness.values}')


# Define Bayesian optimization function


def run_bayesian_optimization():
    res = gp_minimize(evaluate_agent,  # the function to minimize
                      [(0.0001, 0.1), (10, 100)],  # the bounds on each dimension of x
                      acq_func="EI",  # the acquisition function
                      n_calls=10,  # the number of evaluations of f
                      n_random_starts=5,  # the number of random initialization points
                      random_state=1234)  # the random seed
    print("Best parameters found: ", res.x)
    print("Best value found: ", res.fun)
    plot_convergence(res)


# Running optimization methods
run_bayesian_optimization()
run_genetic_algorithm()
