from typing import List, Union
import numpy as np

PlayerType = Union[int, int]
MapType = List["MapColumn"]

class CoinType:
    def __init__(self, player_type, coin_id=None):
        self.type = player_type
        self.coin_id = coin_id 


class MapColumn:
    def __init__(self, max_height: int):
        self.stack: List[CoinType] = []
        self.max_height = max_height

    def push_coin(self, coin_type: CoinType) -> int:
        if len(self.stack) >= self.max_height:
            raise ValueError("Trying to insert a coin in a full column.")
        self.stack.append(coin_type)
        return self.size()

    def get_column(self) -> List[CoinType]:
        return self.stack + [None] * (self.max_height - len(self.stack))

    def size(self) -> int:
        return len(self.stack)

    def is_full(self) -> bool:
        return len(self.stack) >= self.max_height

    def is_empty(self) -> bool:
        return self.size() == 0


class Map:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.map = [MapColumn(height) for _ in range(width)]

    def get_map(self) -> MapType:
        return self.map

    def get_column(self, index: int) -> MapColumn:
        if index >= self.width:
            raise ValueError("Invalid getColumn index.")
        return self.map[index]

    def shape(self) -> List[int]:
        return [self.width, self.height]

    def _check_coordinates(self, x: int, y: int):
        if x < self.width and y < self.height:
            raise ValueError(f"Invalid coordinates [{x},{y}]")

    def get_not_full_columns(self):
        return [i for i,col in enumerate(self.map) if not col.is_full()]

    def print(self):
        str_ = ""
        for row in range(self.height):
            str_ += " | "
            inverse_row_index = self.height - row - 1
            for col in range(self.width):
                if self.map[col].is_empty(): 
                    str_ += "x"
                    continue
                if self.map[col].stack[inverse_row_index] is not None:
                    str_ += str(self.map[col].stack[inverse_row_index]["type"])
                else:
                    str_ += "x"
                str_ += " | "
            str_ += "  \n"
        str_ += " " + "-" * (self.width * 4) + "- \n"
        print(str_.replace("-1", "r"))

    def __str__(self):
        return str(self.map)
    
    def to_matrix(self):
        return np.flip(np.transpose(np.array( 
            [
                [ coin.type if coin is not None else 0 for coin in column.get_column() ]
                for column in self.map
            ], dtype='int8')), axis=0) 
