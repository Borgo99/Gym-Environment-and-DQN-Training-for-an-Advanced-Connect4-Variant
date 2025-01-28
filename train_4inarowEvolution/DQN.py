import gym
import numpy as np
import pandas as pd
import random, os, shutil, time
from collections import deque
from tqdm import tqdm
from typing import Literal
# Tensorflow framework
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import Four in a Row - Evolution environment
from .FourInARowEnv import FourInARowEnv
from .game import Game
from .game_map import Map


class DQN:
  """
  A Deep Q-Network (DQN) class for training and evaluating an agent in a "Four-in-a-Row Evolution" environment.
  This class includes methods to define models, train the agent, and evaluate its performance.
  """

  def __init__(self,
               state_space,
               action_space,
               env,
               gamma=0.95):
    """
    Initialize the DQN agent.
    Args:
    - state_space: The dimensions of the state space (e.g., the game board).
    - action_space: The number of possible actions.
    - env: The environment in which the agent operates.
    - gamma: Discount factor for future rewards.
    """
    self.state_space = state_space
    self.action_space = action_space
    self.env = env
    self.gamma = gamma
    self.model = None
    self.target_model = None
    self.best_model = None
    self.current_training_config = {}

  def create_model(self,
                   model,
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                   loss="mse",
                   metrics=["accuracy"]):       
    """
    Create and compile the DQN model and initialize the target and best models.
    Args:
    - model: The neural network model architecture.
    - optimizer: The optimizer for training.
    - loss: Loss function.
    - metrics: Metrics to monitor during training.
    """
    self.model = tf.keras.models.clone_model(model)
    self.model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics
    )
    self.target_model = tf.keras.models.clone_model(model)
    self.target_model.set_weights(self.model.get_weights())
    self.best_model = keras.models.clone_model(self.model)
    self.best_model.set_weights(self.model.get_weights())

  def load_model(self, model_path):
    """
    Load a pre-trained model from a file.
    Args:
    - model_path: Path to the saved model file.
    """
    self.model = keras.models.load_model(model_path, safe_mode=False)
    self.target_model = keras.models.clone_model(self.model)
    self.target_model.set_weights(self.model.get_weights())
    self.best_model = keras.models.clone_model(self.model)
    self.best_model.set_weights(self.model.get_weights())

  def save_model(self, path):
    """
    Save the current model to a file.
    Args:
    - path: The path to save the model.
    """
    self.model.save(path)

  def get_action(self,
                 state,
                 model,
                 max_random_skips=0):
    """
    Select an action based on the Q-values predicted by the model.
    Args:
    - state: The current state of the environment.
    - model: The model used to predict Q-values.
    - max_random_skips: Maximum number of top actions to skip randomly (enhance exploration, default 0).
    Returns:
    - action: The selected action.
    - action_index: The index of the selected action.
    """
    Q_values = model(state.reshape(1, *self.state_space),
                     training=False)[0].numpy()
    actions_to_skip = np.random.randint(0, max_random_skips + 1)
    
    for action_index in np.argsort(Q_values)[::-1]:
      action = self.env.action_set[action_index]
      if self.env.is_action_legal(action):
        if actions_to_skip > 0:
          actions_to_skip -= 1
        else:
          return action, action_index
    return action, action_index

  def update_network(self, batch):
    """
    Train the DQN model using a batch of experiences from the replay buffer.
    Args:
    - batch: A batch of experiences (state, action, reward, next_state, done).
    Returns:
    - history: The training history object returned by TensorFlow fit method.
    """
    batch_size = len(batch)
    state_batch = np.zeros(shape=(batch_size, *self.state_space))
    next_state_batch = np.zeros(shape=(batch_size, *self.state_space))

    for i, (state, _, _, next_state, _) in enumerate(batch):
      state_batch[i, :] = state
      next_state_batch[i, :] = next_state

    Q_values_state = self.model(state_batch)
    Q_values_next_state = self.target_model(next_state_batch)

    y_train = np.zeros((batch_size, self.action_space))
    for i, (_, action, reward, _, done) in enumerate(batch):
      y_train[i, :] = Q_values_state[i]
      y_train[i, action] = reward if done else reward + self.gamma * np.max(Q_values_next_state[i])

    history = self.model.fit(
      state_batch,
      y_train,
      batch_size = batch_size,
      verbose = 0,
      epochs = 1
    )
    return history

  def compute_win_rate_vs_random_opponent(self, episodes=500, use_heuristic=True):
    """
    Evaluate the agent's win rate against a random opponent.
    Args:
    - episodes: Number of episodes to play.
    - use_heuristic: If True, the opponent uses a heuristic strategy (default True).
    Returns:
    - win_rate: Win rate of the agent.
    - tie_rate: Tie rate of the agent.
    - action_count: Count of how many times each action has been taken.
    """
    win_count = 0
    tie_count = 0
    action_count = {"".join(map(str, a)): 0 for a in self.env.action_set}
    for _ in range(episodes):
      state = self.env.reset()
      episode_terminated = False
      while not episode_terminated:
        action, action_index = self.get_action(state, model=self.model)
        opponent_action = None
        if not use_heuristic:
          opponent_action = self.env.action_set[np.random.randint(0, len(self.env.action_set))]
        new_state, reward, episode_terminated, _ = self.env.step(
          action,
          reward_when_episode_ends=True,
          opponent_action=opponent_action)
        state = new_state
        action_count["".join(map(str, action))] += 1
      if reward > 0:
        win_count += 1
      elif reward == 0:
        tie_count += 1

    return round(win_count/episodes, 3), round(tie_count/episodes, 3), action_count
  
  def explore(self, eps):
    """
    Perform exploration with a probability epsilon (eps).
    Args:
    - eps: The exploration rate.
    Returns:
    - action: The randomly selected action.
    - action_index: The index of the action.
    """
    if np.random.random() < eps:
      while True:
        action_index = np.random.randint(self.action_space)
        action = self.env.action_set[action_index]
        if self.env.is_action_legal(action):
          return action, action_index
    else: return None, None
        
  def episode_step(self, state, eps):
    """
    Perform a single step in the episode.
    Args:
    - state: The current state.
    - eps: The exploration rate.
    Returns:
    - action: The agent's action.
    - action_index: The index of the action.
    - result: Tuple (new_state, reward, episode_terminated).
    """
    action = None
    # Exploring
    action, action_index = self.explore(eps=eps)
    # Exploiting
    if action is None:
      action, action_index = self.get_action(
        state, 
        model=self.model,
        max_random_skips=self.current_training_config["max_random_skips"])
    # Opponent action
    strategy = self.current_training_config["strategy"]
    if strategy == "mixed":
      v = np.random.random()
      if v < 0.3:
        strategy = "random"
      elif v < 0.55:
        strategy = "target"
      elif v < 0.8:
        opponent_action, opponent_action_index = self.get_action(state, model=self.best_model)
      else:
        opponent_action, opponent_action_index = self.get_action(state, model=self.model)

    if strategy == "random":
      opponent_action = None
    elif strategy == "target":
      opponent_action, opponent_action_index = self.get_action(state, model=self.target_model)
    # Run action in env
    new_state, reward, episode_terminated, _ = self.env.step(
      action,
      reward_when_episode_ends=True,
      opponent_action=opponent_action)
    return action, action_index, (new_state, reward, episode_terminated)
  
  def run_episode(self, eps):
    """
    Run a full episode with exploration and exploitation.
    Args:
    - eps: Exploration rate.
    Returns:
    - episode_history: List of experiences from the episode (state, action, reward, next_state, done).
    """
    state = self.env.reset()
    episode_terminated = False
    episode_history = []
    while not episode_terminated:
      action, action_index, (new_state, reward, episode_terminated) = self.episode_step(state, eps)
      # Store step results
      episode_history.append((state, action_index, reward, new_state, episode_terminated))
      state = new_state
    return episode_history

  def train(self,
            episodes=10_000,
            buffer_size=100_000,
            min_episodes_before_eps_update=5_000,
            eps_start=1.0,
            eps_end=0,
            episodes_after_eps_end=0,
            max_random_skips=0,
            target_update_interval=1_000,
            strategy: Literal["random", "target", "mixed"] = "target",
            batch_size=64,
            min_buffer_size_before_training=64*10,
            network_update_per_episode=1,
            evaluate_vs_random_interval=3_000,
            warmup_episodes=0,
            best_win_rate_vs_random=0.0,
            timeout=3*60*60,
            checkpoints_pathname="./model_checkpoints"):
    """
    Train the DQN agent.
    Args:
    - episodes: Total episodes for training.
    - buffer_size: Maximum size of the replay buffer.
    - min_episodes_before_eps_update: Minimum episodes before decreasing epsilon.
    - eps_start: Starting epsilon value.
    - eps_end: Minimum epsilon value.
    - episodes_after_eps_end: Episodes after reaching minimum epsilon.
    - max_random_skips: Maximum actions to skip randomly during exploitation.
    - target_update_interval: Steps between target network updates.
    - strategy: Opponent strategy
      - "random": play always against the random-heuristic opponent
      - "target": play always against the target model
      - "mixed": play randomly against random-heuristic, target model, current model and best model
    - batch_size: Batch size for training.
    - min_buffer_size_before_training: Minimum replay buffer size before training starts.
    - network_update_per_episode: Number of training steps per episode.
    - evaluate_vs_random_interval: Interval for evaluating against random opponents.
    - warmup_episodes: Initial episodes for populating replay buffer. Each episode is simulated with an eps value randomly sample between 1.0 and eps_start.
    - best_win_rate_vs_random: Best win rate achieved during training.
    - timeout: Maximum training time in seconds.
    - checkpoints_pathname: Path for saving model checkpoints.
    Returns:
    - success: Boolean indicating whether training has completed or has timeout.
    - training_summary: Dictionary with training statistics:
      - "eps": last eps value 
      - "episodes_done": episodes simulated 
      - "best_win_rate_vs_random": best score attained
      - "win_rate_history": list of win rates achieved at each evaluation step
      - "last_model_action_count_vs_random": action_count from evaluation step
    """
            
    if self.model is None: raise ValueError("self.model is None")

    self.current_training_config["max_random_skips"] = max_random_skips
    self.current_training_config["strategy"] = strategy

    if not os.path.exists(checkpoints_pathname):
      os.mkdir(checkpoints_pathname)

    replay_buffer = deque(maxlen=buffer_size)
    eps = eps_start
    train_step_count = 0
    start_time = time.time()
    win_rate_history = []
    eps_decay = (eps_start-eps_end)/(episodes - min_episodes_before_eps_update - episodes_after_eps_end)

    print(f"{warmup_episodes=}")
    for episode in tqdm(range(warmup_episodes)):
      episode_history = self.run_episode(eps=random.uniform(eps, 1.0))
      # Cap reward
      reward = 1 if episode_history[-1][2] > 0 else -1
      # Update each step of the episode with the final reward
      for state, action_index, _, new_state, done in episode_history:
        replay_buffer.append((state, action_index, reward, new_state, done))
    print(f"Initial Replay Buffer size: {len(replay_buffer)}")

    print("\n * Start training * \n")
    for episode in tqdm(range(episodes)):
      if (time.time() - start_time) > timeout:
        # Store trained model weights
        self.save_model(f"{checkpoints_pathname}/trained_model.keras")
        win_rate, tie_rate, _ = self.compute_win_rate_vs_random_opponent()
        print(f"\n * Timeout * \nLast model Win rate vs random opponent: {win_rate*100:.1f}%")
        return False, {
          "eps": eps, "episodes_done": episode-1, 
          "best_win_rate_vs_random": best_win_rate_vs_random,
          "win_rate_history": win_rate_history}

      episode_history = self.run_episode(eps)
      # Cap reward
      reward = 1 if episode_history[-1][2] > 0 else -1
      # Update each step of the episode with the final reward
      for state, action_index, _, new_state, done in episode_history:
        replay_buffer.append((state, action_index, reward, new_state, done))

      # Update exploration eps
      if episode > min_episodes_before_eps_update and eps > eps_end:
        eps -= eps_decay

      # Train step
      if len(replay_buffer) < min_buffer_size_before_training:
        continue

      for _ in range(network_update_per_episode):
        batch = random.sample(replay_buffer, batch_size)
        history = self.update_network(batch)
        train_step_count += 1

        # Update target network
        if (train_step_count % target_update_interval) == 0:
          self.target_model.set_weights(self.model.get_weights())

        # Calculate win rate vs random opponent
        if (train_step_count % evaluate_vs_random_interval) == 0:
          win_rate, tie_rate, _ = self.compute_win_rate_vs_random_opponent()
          print(f'\r * Train step {train_step_count} ✅ *', end='\n')
          print(f'Epsilon value: {eps:.4f}')
          print(f'Loss: {history.history["loss"][-1]}')
          print(f'Accuracy: {history.history["accuracy"][-1]}')
          print(f"Win rate vs random opponent: {win_rate*100}% ({tie_rate=})")
          print(f"Replay Buffer size: {len(replay_buffer)}/{buffer_size}")
          win_rate_history.append({"win_rate": win_rate, "episode": episode-1})

          if win_rate > best_win_rate_vs_random:
            best_win_rate_vs_random = win_rate
            self.best_model.set_weights(self.model.get_weights())
            self.save_model(f"{checkpoints_pathname}/best.keras")
            print(f"Best weights updated at episode {episode}")

    # Store trained model weights
    self.save_model(f"{checkpoints_pathname}/trained_model.keras")
    win_rate, tie_rate, action_count = self.compute_win_rate_vs_random_opponent(episodes=2_000)
    print(f"Last model Win rate vs random opponent: {win_rate*100:.1f}% ({tie_rate=})")
    win_rate_history.append({"win_rate": win_rate, "episode": episode-1})
    return True, {
      "eps": eps, 
      "episodes_done": episode-1, 
      "best_win_rate_vs_random": best_win_rate_vs_random,
      "win_rate_history": win_rate_history,
      "last_model_action_count_vs_random": action_count}
