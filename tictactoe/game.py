import numpy as np
import os


def check_sequence(sequence: np.ndarray):
    if np.all(sequence == 1) or np.all(sequence == 2):
        return True, 1 if np.all(sequence == 1) else 2
    return False, 0


class TicTacToe:
    rows = 3
    columns = 3

    def __init__(self):
        self.board = np.zeros(self.rows * self.columns, dtype=int)

    def get_board(self, reshape=True):
        if not reshape:
            return self.board
        return self.board.reshape((3, 3))

    def play_move(self, row: int, column: int, mark: str):
        if not (0 <= row < self.rows and 0 <= column < self.columns):
            raise ValueError("Row or Column is out of range")
        reshaped_board = self.get_board()
        if reshaped_board[row][column] == 0:
            reshaped_board[row][column] = mark
        else:
            raise ValueError("Cell is already occupied")
        self.board = reshaped_board.reshape((self.rows * self.columns))

    # For checking if any player is winning
    def is_terminal(self):
        reshaped_board = self.get_board()

        # Check Horizontal Axes
        for row in range(self.rows):
            inarow, winner = check_sequence(reshaped_board[row])
            if inarow:
                return True, winner

        # Check Vertical
        for col in range(self.columns):
            inarow, winner = check_sequence(reshaped_board[:, col:col + 1])
            if inarow:
                return True, winner

        # Check Positive Diagonal
        inarow, winner = check_sequence(reshaped_board.diagonal())
        if inarow:
            return True, winner

        # Check Negative Diagonal
        inarow, winner = check_sequence(reshaped_board[::-1].diagonal())
        if inarow:
            return True, winner

        # Check if board is full
        if not np.any(reshaped_board == 0):
            return True, 0

        return False, 0


#
# game.play_move(0, 1, 1)
# game.play_move(1, 1, 2)
# game.play_move(1, 0, 2)
# game.play_move(1, 2, 2)
# game.play_move(2, 1, 1)
# print(game.get_board())
# print(game.is_terminal())

game = TicTacToe()
while True:
    rows, cols = game.get_board().shape
    for row in range(rows):
        for col in range(cols):
            print(game.get_board()[row][col], end=" ")
        print("")

    terminal, winner = game.is_terminal()
    if terminal:
        if winner != 0:
            print("You Won") if winner == 1 else print("You Lost")
        break

    move_row, move_col = input("enter you move (row col): ").split(" ")
    try:
        game.play_move(int(move_row), int(move_col), 1)
    except ValueError as e:
        print(e)
    os.system("cls")
