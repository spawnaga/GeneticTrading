# import gymnasium as gym
# import numpy as np
# from gymnasium.envs.registration import register
#
# # Register the environment once, if not already registered, to avoid redundant registration warnings
# if "stocks-v0" not in gym.envs.registry:
#     register(
#         id='stocks-v0',
#         entry_point='gym_anytrading.envs:StocksEnv',
#     )
#
# class TradingEnvWrapper:
#     """
#     A wrapper around gym-anytrading's trading environment, designed for compatibility with distributed
#     reinforcement learning training. This wrapper computes the flattened observation dimension and
#     incorporates percentage-based commissions and fees into the reward calculation.
#
#     Attributes:
#         env (gym.Env): The underlying gym-anytrading environment (e.g., StocksEnv).
#         observation_space (gym.spaces.Box): The observation space of the environment.
#         action_space (gym.spaces.Discrete): The action space of the environment (Discrete(2) for Buy/Sell).
#         observation_dim (int): The flattened dimension of the observation space for use in RL algorithms.
#         commission_percent (float): Percentage commission applied per trade.
#         fee_percent (float): Percentage fee applied per trade.
#     """
#
#     def __init__(
#         self,
#         df,
#         window_size=10,
#         frame_bound=None,
#         env_type='stocks-v0',
#         commission_percent=0.001,  # 0.1% commission by default
#         fee_percent=0.0005         # 0.05% fee by default
#     ):
#         """
#         Initialize the trading environment with a data chunk and trading cost parameters.
#
#         Args:
#             df (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', 'Close' columns.
#             window_size (int): Number of past time steps included in each observation.
#             frame_bound (tuple): (start, end) indices defining the data range; defaults to full DataFrame.
#             env_type (str): Type of environment ('stocks-v0' or 'forex-v0').
#             commission_percent (float): Percentage commission per trade (e.g., 0.001 = 0.1%).
#             fee_percent (float): Percentage fee per trade (e.g., 0.0005 = 0.05%).
#         """
#         # Set default frame_bound to use the full DataFrame if not provided
#         if frame_bound is None:
#             frame_bound = (window_size, len(df))
#
#         # Create the underlying trading environment
#         self.env = gym.make(
#             env_type,
#             df=df,
#             window_size=window_size,
#             frame_bound=frame_bound,
#             disable_env_checker=True  # Disable checker to prevent action space compatibility issues
#         )
#
#         # Store observation and action spaces
#         self.observation_space = self.env.observation_space  # Shape: (window_size, num_features)
#         self.action_space = self.env.action_space            # Discrete(2) for Buy/Sell
#
#         # Store trading cost parameters
#         self.commission_percent = commission_percent
#         self.fee_percent = fee_percent
#
#         # Compute the flattened observation dimension for compatibility with RL algorithms
#         if isinstance(self.observation_space, gym.spaces.Box):
#             self.observation_dim = int(np.prod(self.observation_space.shape))
#         else:
#             raise ValueError("Observation space must be a Box for this implementation.")
#
#     def reset(self):
#         """
#         Reset the environment to its initial state.
#
#         Returns:
#             np.ndarray: Initial observation as a 2D array (shape: (window_size, num_features)).
#         """
#         obs = self.env.reset()
#         return obs
#
#     def step(self, action):
#         """
#         Execute one step in the environment, applying percentage-based commissions and fees to the reward.
#
#         Args:
#             action (int): Action to take (0 = Sell, 1 = Buy; assumes 0 may represent Hold in some contexts).
#
#         Returns:
#             tuple: (observation, reward, done, info)
#                 - observation (np.ndarray): Next observation (2D array).
#                 - reward (float): Reward adjusted for trading costs.
#                 - done (bool): Whether the episode has ended.
#                 - info (dict): Additional information from the environment.
#         """
#         # Perform the step in the underlying environment
#         obs, reward, done, info = self.env.step(action)
#
#         # Apply trading costs (commission + fee) to the reward when a trade occurs
#         trade_cost = self.commission_percent + self.fee_percent
#         if action != 0:  # Apply costs only for Buy (1) or Sell (non-zero), assuming 0 is Hold
#             reward -= trade_cost * abs(reward)  # Reduce reward proportional to its magnitude
#
#         return obs, reward, done, info
#
#     def render(self, mode='human'):
#         """
#         Render the current state of the environment.
#
#         Args:
#             mode (str): Rendering mode ('human' or other modes supported by gym-anytrading).
#
#         Returns:
#             The rendered output (format depends on mode).
#         """
#         return self.env.render(mode=mode)
#
#     def render_all(self):
#         """
#         Render all trades in the episode, typically used for post-episode visualization.
#
#         Returns:
#             The complete rendered output of all trades.
#         """
#         return self.env.render_all()
#
#     def close(self):
#         """
#         Close the environment and release any resources.
#         """
#         return self.env.close()
#
#     @property
#     def done(self):
#         """
#         Check if the current episode has finished.
#
#         Returns:
#             bool: True if the episode is complete, False otherwise.
#         """
#         return self.env.unwrapped._done
