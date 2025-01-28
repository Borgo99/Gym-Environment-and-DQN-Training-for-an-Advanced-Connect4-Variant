import numpy as np
from .DQN import DQN


class DoubleDQN(DQN):
  """
  A Double Deep Q-Network (Double DQN) class for training and evaluating an agent in a "Four-in-a-Row Evolution" environment.
  The Double DQN mitigates the overestimation bias in Q-values by decoupling the action selection and evaluation processes.

  Inherits from the DQN class and overrides the network update mechanism.
  """

  def __init__(self, state_space, action_space, env, gamma=0.95):
    """
    Initialize the Double DQN agent.
    Args:
    - state_space: The dimensions of the state space (e.g., the game board).
    - action_space: The number of possible actions.
    - env: The environment in which the agent operates.
    - gamma: Discount factor for future rewards.
    """
    super().__init__(state_space, action_space, env, gamma)

  def update_network(self, batch):
    """
    Train the Double DQN model using a batch of experiences from the replay buffer.
    This method uses the Double DQN approach to calculate target Q-values.

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

    Q_values_state_online = self.model(state_batch)
    Q_values_next_state_online = self.model(next_state_batch)
    Q_values_next_state_target = self.target_model(next_state_batch)

    y_train = np.zeros((batch_size, self.action_space))
    for i, (_, action, reward, _, done) in enumerate(batch):
      y_train[i, :] = Q_values_state_online[i]
      if done:
        y_train[i, action] = reward
      else:
        best_next_state_action = np.argmax(Q_values_next_state_online[i])
        target_Q_value = Q_values_next_state_target[i, best_next_state_action]
        y_train[i, action] = reward + self.gamma * target_Q_value
        
    history = self.model.fit(
      state_batch,
      y_train,
      batch_size = batch_size,
      verbose = 0,
      epochs = 1
    )
    return history