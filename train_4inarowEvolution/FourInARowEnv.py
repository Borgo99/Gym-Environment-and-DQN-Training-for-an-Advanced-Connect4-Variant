import gym
from gym import spaces
import numpy as np
import random
from .game import Game
from .game_map import Map

class FourInARowEnv(gym.Env):
    """
    A custom Gym environment for the 4 in a Row - Evolution game.
    Manages the game state, actions, rewards, and observations.
    """
    
    def __init__(self, map_width=7, map_height=6, moves_at_time=3, rounds=None, render_mode='human'):
        """
        Initialize the environment.
        Args:
        - map_width: The width of the game board.
        - map_height: The height of the game board.
        - moves_at_time: Number of moves each player can make per turn.
        - rounds: Number of rounds in the game. If None, calculated based on the board size.
        - render_mode: Mode for rendering the game (default is 'human').
        """
        super(FourInARowEnv, self).__init__()
        self.render_mode = render_mode
        
        self.width = map_width
        self.height = map_height
        self.ai_color = 'red' # ! 1 == yellow and -1 == red
        self.ai_type = -1
        self.opponent_type = 1
        self.opponent_color = 'yellow'
        self.opponent_move = lambda: np.random.randint(0, self.width)
        
        self.moves_at_time = moves_at_time
        self.rounds = rounds
        self.game = Game(Map(self.width, self.height), moves_at_time=self.moves_at_time, rounds=self.rounds)
        self.action_space = spaces.MultiDiscrete([self.width]*self.moves_at_time, dtype=np.int8)
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(self.height, self.width),
            dtype=np.int8)
        
        columnIndexes = np.arange(map_width)
        meshgrid = np.meshgrid(*[columnIndexes]*moves_at_time)
        self.action_set = np.stack(meshgrid, axis=-1).reshape(-1, moves_at_time)
        
    def reset(self):
        """
        Reset the environment to start a new game.
        Returns:
        - obs: Initial board state as an observation.
        """
        self.game = Game(Map(self.width, self.height), self.moves_at_time, self.rounds)
        self.current_step = 0
        return self._next_observation()
    
    def _next_observation(self):
        """
        Get the current state of the game board as an observation.
        Returns:
        - obs: A matrix representing the board state.
        """
        obs = self.game.map.to_matrix()
        return obs
    
    def step(self, action, reward_when_episode_ends = False, opponent_action=None):
        """
        Executes the player’s action and the opponent’s move. 
        The opponent plays first and follows a combination of random and heuristic logic 
        Args:
        - action: list of columns indices.
        - reward_when_episode_ends: Whether to compute reward only at the end of the game or at each round.
        - opponent_action: Optional opponent moves (default is None, uses a heuristic).
        Returns:
        - obs: The new board state.
        - reward: The reward for the step.
        - done: Whether the game is over.
        - info: Additional info (empty dict in this case).
        """
        scores = self.game.get_scores()
        ai_score = scores[self.ai_color]
        opponent_score = scores[self.opponent_color]
        diff_before = ai_score - opponent_score

        self._take_action(action, reward_when_episode_ends, opponent_action)

        scores = self.game.get_scores()
        ai_score = scores[self.ai_color]
        opponent_score = scores[self.opponent_color]
        diff_after = ai_score - opponent_score
        
        self.current_step += 1

        # reward is zero if reward_when_episode_ends is True
        reward = diff_after - diff_before
        done = self.game.is_game_over()
        if done:
            obs = self._next_observation()
            #self.reset()
        else:
            obs = self._next_observation()
        return obs, reward, done, {}
    
    def _take_action(self, action, reward_when_episode_ends = False, opponent_action=None):
        """
        Action step logic. Use the step method instead.
        """
        # Opponent action is played before
        if opponent_action is None:
            opponent_action = self.game.get_hint()
        for i,ai_move in enumerate(action):
            opp_coin = self.game.get_coin(self.opponent_type)
            self.game.push_coin(opponent_action[i], opp_coin)
            # user move
            self.game.push_coin(ai_move, self.game.get_coin(self.ai_type))
        
        if not reward_when_episode_ends:
            self.game.find_4_rows() # run at each round
        elif self.game.rounds_played == self.game.rounds - 1: 
            self.game.find_4_rows() # run only at the last round
        else: self.game.rounds_played += 1
            
    def render(self, mode='human', close=False):
        """
        Render the current game state to the console.
        Args:
        - mode: Render mode (default is 'human').
        - close: Whether to close the rendering (not used here).
        """
        print(f'Step: {self.current_step}')
        print(f'Current score: {self.game.get_scores()}')
        print(f'Current score AI: {self.game.get_scores()[self.ai_color]}')

    def available_actions(self):
        """
        Get the list of columns that are not full.
        Returns:
        - List[int]: List of column indexes with available space.
        """
        return self.game.map.get_not_full_columns()

    def is_action_legal(self, action):
        """
        Check if a given action is legal.
        Args:
        - action: List of column indexes to validate.
        Returns:
        - bool: True if the action is legal, False otherwise.
        """
        legal_actions = set(self.available_actions())
        return all(sub_action in legal_actions for sub_action in action)