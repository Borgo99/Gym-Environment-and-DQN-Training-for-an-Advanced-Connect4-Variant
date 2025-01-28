from typing import List, Tuple, Set, Union, Dict
from .game_map import CoinType, Map, PlayerType
import numpy as np


class Game:
    """
    A class representing the game logic for 4 in a Row - Evolution. Handles the game board, moves, 
    scoring, and checking for winning conditions.
    """

    def __init__(
        self,
        map: Map,
        moves_at_time: int = 3,
        rounds: int = None,
        stop_at_rounds: bool = True
    ):
        """
        Initialize the game with the provided map, number of moves per turn, and total rounds.
        Args:
        - map: An instance of the Map class representing the game board.
        - moves_at_time: Number of moves each player can make in a turn.
        - rounds: Total number of rounds to play. If None, it's calculated based on the board size.
        - stop_at_rounds: Whether to stop the game after the specified number of rounds (default True).
        Raises:
        - ValueError: If the map size, number of rounds, or moves are invalid.
        """
        self.map = map
        self.width, self.height = map.shape()
        if self.width < 3 or self.height < 3:
            raise ValueError("Invalid map shape: too small.")
        rounds_to_fill_map = (self.width*self.height)/(moves_at_time*2)
        if rounds is None:
            if not rounds_to_fill_map.is_integer():
                raise ValueError(f"This Map shape cannot work with {moves_at_time=}")
            self.rounds = rounds_to_fill_map
        elif rounds<rounds_to_fill_map:
            self.rounds = rounds
        else: raise ValueError('Invalid rounds number for this Map shape.')
            
        self.stop_at_rounds = stop_at_rounds
        self.moves_at_time = moves_at_time
        max_number_of_moves = moves_at_time * 2 * self.rounds
        if self.width * self.height < max_number_of_moves:
            raise ValueError(
                "Invalid Game input: there are too many possible moves with respect to Map shape."
            )
        self.rounds_played = 0
        self.yellow_scores = set()
        self.red_scores = set()
        self.coins_counter = 0

    def get_coin(self, type: PlayerType) -> CoinType:
        """
        Generate a new coin for a player.
        Args:
        - type: Player type (1 for yellow, -1 for red).
        Returns:
        - CoinType: A coin with the player's type and a unique ID.
        """
        self.coins_counter += 1
        return CoinType( type, self.coins_counter)

    def push_coin(self, column: int, coin: CoinType) -> Tuple[int, int]:
        """
        Place a coin in the specified column or find the nearest available column.
        Args:
        - column: The target column for the coin.
        - coin: The coin to place.
        Returns:
        - Tuple[int, int]: The (row, column) where the coin was placed.
        Raises:
        - ValueError: If the map is full or the coin ID is invalid.
        """
        if coin.coin_id < 1:
            raise ValueError("CoinId must be greater than 0.")
        if not self.map.get_column(column).is_full():
            return (
                self.map.get_map()[column].push_coin(coin) - 1,
                column,
            )
        left_columns = [column - 1 - i for i in range(column)]
        right_columns = [i + column + 1 for i in range(self.width - column - 1)]
        for index in range(max(len(left_columns), len(right_columns))):
            if (
                len(left_columns) > index
                and not self.map.get_column(left_columns[index]).is_full()
            ):
                return (
                    self.map.get_map()[left_columns[index]].push_coin(coin) - 1,
                    left_columns[index],
                )
            if (
                len(right_columns) > index
                and not self.map.get_column(right_columns[index]).is_full()
            ):
                return (
                    self.map.get_map()[right_columns[index]].push_coin(coin) - 1,
                    right_columns[index],
                )
        raise ValueError("Map is full: each column is at capacity.")

    def get_line_id(self, cells_id: List[int]) -> int:
        """
        Generate a unique ID for a line of connected coins.
        Args:
        - cells_id: List of coin IDs.
        Returns:
        - int: A unique ID for the line.
        """
        return int("".join(str(cell_id).zfill(2) for cell_id in cells_id))

    def get_cells_from_line_id(self, line_id: int) -> List[int]:
        """
        Extract coin IDs from a line ID.
        Args:
        - line_id: The unique ID of the line.
        Returns:
        - List[int]: The list of coin IDs in the line.
        """
        line_id_str = str(line_id).zfill(8)
        return [int(cell) for cell in line_id_str]

    def find_4_rows(self) -> List[int]:
        """
        Find all new 4-in-a-row lines - vertically, horizontally and diagonally - on the board.
        Returns:
        - List[int]: List of line IDs representing new 4-in-a-row lines.
        Raises:
        - ValueError: If the game is over or terminated.
        """
        if self.stop_at_rounds and self.rounds_played >= self.rounds:
            raise ValueError("Invalid find4Rows call: game is terminated.")
        new_lines_found = []
        # Check vertical 4-in-a-row
        for column in self.map.get_map():
            if column.size() < 4:
                continue

            cumulative_value = 0
            row_ids = []

            for i, cell in enumerate(column.get_column()):
                if cell is None: break
                if i == 0:
                    # First cell
                    cumulative_value = cell.type
                    row_ids = [cell.coin_id]
                    continue

                # Cell of different color
                if cumulative_value * cell.type < 0:
                    # Line interrupted
                    if column.size() - i < 4:
                        break
                    cumulative_value = cell.type
                    row_ids = [cell.coin_id]
                    continue

                # Cell of the same color
                row_ids.append(cell.coin_id)

                if abs(cumulative_value) == 3:
                    # 4-in-a-row found
                    line_id = self.get_line_id(row_ids)
                    row_ids = row_ids[1:]

                    if cumulative_value > 0:
                        if line_id in self.yellow_scores:
                            continue
                        self.yellow_scores.add(line_id)
                    else:
                        if line_id in self.red_scores:
                            continue
                        self.red_scores.add(line_id)

                    new_lines_found.append(line_id)
                else:
                    cumulative_value += cell.type

        self.check_lines_with_cumulative_values(new_lines_found, "horizontal")
        self.check_lines_with_cumulative_values(new_lines_found, "rightDiagonal")
        self.check_lines_with_cumulative_values(new_lines_found, "leftDiagonal")

        self.rounds_played += 1

        return new_lines_found

    def check_lines_with_cumulative_values(self, new_lines_found: List[int], type: str):
        """
        Check for 4-in-a-row in a specific direction.
        Args:
        - new_lines_found: List to store new line IDs.
        - type: Direction to check ("horizontal", "rightDiagonal", "leftDiagonal").
        """
        col_length = self.height if type == "horizontal" else self.width - 3 + self.height - 4
        cumulative_value = [0] * col_length
        row_ids = [[] for _ in range(col_length)]
        map = self.map.get_map() if type != "leftDiagonal" else self.map.get_map().copy()[::-1]

        for col_index, column in enumerate(map):
            if column.is_empty():
                cumulative_value = [0] * col_length
                row_ids = [[] for _ in range(col_length)]
                continue

            skip_to_row = max(4 - self.width + col_index, 0)
            cells = column.get_column()[skip_to_row:self.height - 3 + col_index] if type != "horizontal" else column.get_column()

            for row_index, cell in enumerate(cells):
                if cell is None: break
                index = row_index if type == "horizontal" else row_index + skip_to_row - col_index + self.width - 4

                if cumulative_value[index] == 0:
                    cumulative_value[index] = cell.type
                    row_ids[index].append(cell.coin_id)
                    continue

                if cumulative_value[index] * cell.type < 0:
                    cumulative_value[index] = cell.type
                    row_ids[index] = [cell.coin_id]
                    continue

                row_ids[index].append(cell.coin_id)

                if abs(cumulative_value[index]) == 3:
                    line_id = self.get_line_id(row_ids[index])
                    row_ids[index] = row_ids[index][1:]

                    if cumulative_value[index] > 0:
                        if line_id in self.yellow_scores:
                            continue
                        self.yellow_scores.add(line_id)
                    else:
                        if line_id in self.red_scores:
                            continue
                        self.red_scores.add(line_id)

                    new_lines_found.append(line_id)
                else:
                    cumulative_value[index] += cell.type

            if column.size() < self.height:
                for i in range(self.height - column.size()):
                    row_index = i + column.size()
                    index = row_index if type == "horizontal" else row_index - col_index + self.width - 4
                    if index >= len(cumulative_value): break
                    cumulative_value[index] = 0
                    row_ids[index] = []

    def get_scores(self) -> Dict[str, int]:
        """
        Get the current scores for each player.
        Returns:
        - Dict[str, int]: Scores of red and yellow players.
        """
        return {
            'red': len(self.red_scores),
            'yellow': len(self.yellow_scores),
        }

    def get_moves_at_time(self) -> int:
        """
        Get the number of moves allowed per turn.
        Returns:
        - int: Number of moves per turn.
        """
        return self.moves_at_time

    def get_rounds(self) -> int:
        """
        Get the total number of rounds.
        Returns:
        - int: Total rounds.
        """
        return self.rounds

    def is_game_over(self):
        """
        Check if the game is over.
        Returns:
        - bool: True if the game is over, False otherwise.
        """
        return self.rounds <= self.rounds_played

    def get_rounds_left(self) -> int:
        """
        Get the number of rounds left in the game.
        Returns:
        - int: Rounds left.
        """
        return self.rounds - self.rounds_played
    
    def get_hint(self):
        """
        Implements the heurstic logic: provide a hint by identifying strategic columns.
        Returns:
        - List[int]: Suggested columns for the player's next moves.
        """
        moves = []
        # vertical states
        for col_index in range(self.width):
            if (self.map.get_column(col_index).is_full()): continue
            col = [coin.type for coin in self.map.get_column(col_index).get_column() if coin is not None ]
            res = 0
            for v in col[::-1]:
                if v != col[-1]: break
                res += v
            if abs(res) > 2: moves.append(col_index)
        
        while len(moves) < self.moves_at_time:
            moves.append(np.random.randint(0, self.width))

        return moves