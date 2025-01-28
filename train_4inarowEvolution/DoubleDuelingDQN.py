from .DoubleDQN import DoubleDQN
from .DuelingDQN import DuelingDQN

class DoubleDuelingDQN(DoubleDQN, DuelingDQN):
  """
  A Double Dueling Deep Q-Network (Double Dueling DQN) class that combines the benefits of Double DQN and Dueling DQN architectures.
  
  - Double DQN mitigates overestimation bias in Q-values by separating action selection and evaluation.
  - Dueling DQN separates the state-value function and the action-advantage function for better learning efficiency.

  This class inherits from both `DoubleDQN` and `DuelingDQN`.
  """

  def __init__(self, state_space, action_space, env, gamma=0.95):
    """
    Initialize the Double Dueling DQN agent.
    
    Args:
    - state_space: The dimensions of the state space (e.g., the game board).
    - action_space: The number of possible actions.
    - env: The environment in which the agent operates.
    - gamma: Discount factor for future rewards.
    """
    super().__init__(state_space, action_space, env, gamma)