# Importing necessary libraries
import numpy as np
import os


# Function to check if a player has won or not
def check_sequence(sequence: np.ndarray):
    """
    This function takes in an array of integers and checks if all elements are either 1 or 2.
    If so, it returns True with the respective winning player (1 or 2). Otherwise, False with 0 is returned.
    """
    if np.all(sequence == 1) or np.all(sequence == 2):
        return True, 1 if np.all(sequence == 1) else 2
    return False, 0


# Class to represent the game board and its functionality
class TicTacToeBETA:
    """
    This class represents a Tic-Tac-Toe game board with rows and columns set to 3.
    It initializes an empty board and provides methods to interact with the board such as playing a move, checking if anyone has won or if the board is full.
    """
    rows = 3
    columns = 3

    def __init__(self):
        # Initializing the game board as a numpy array of zeros with the same dimensions as Tic-Tac-Toe board
        self.board = np.zeros(self.rows * self.columns, dtype=int)

    def get_board(self, reshape=True):
        """
        This function returns a copy of the game board either in flattened or reshaped (3x3) format depending on the 'reshape' flag.
        If 'reshape' is False, the entire numpy array is returned as it is. Otherwise, the 1D numpy array is reshaped into 2D form (3x3).
        """
        if not reshape:
            return self.board
        return self.board.reshape((3, 3))

    def play_move(self, row: int, column: int, mark: str):
        """
        This function is used to make a move on the game board at the given 'row' and 'column' position with the specified 'mark'.
        If the cell is already occupied or the row and column values are out of range, an exception is raised. Otherwise, the respective cell value in the reshaped (3x3) form of the game board is updated with the 'mark' and then flattened back into 1D.
        """
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
        """
        This function checks the current state of the game board for terminal conditions such as a win or a draw.
        It first reshapes the board to its 2D form and then checks each row, column and diagonal for sequences of all same integers (win) or all cells filled (draw).
        If any such condition is met, it returns True with the respective winning player (1 or 2) if there's a win or False with 0 if there's a draw. Otherwise, it continues to check other conditions.
        """
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


# Game initialization and loop for playing the game
game = TicTacToeBETA()
while True:
    # Displaying the current state of the game board to the player
    rows, cols = game.get_board().shape
    for row in range(rows):
        for col in range(cols):
            print(game.get_board()[row][col], end=" ")
        print("")

    # Checking if any terminal condition has been met
    terminal, winner = game.is_terminal()
    if terminal:
        if winner != 0:
            print("You Won" if winner == 1 else "You Lost")
        break

    # Taking player's move as input and updating the game board accordingly
    move_row, move_col = input("enter you move (row col): ").split(" ")
    try:
        game.play_move(int(move_row), int(move_col), 1)
    except ValueError as e:
        print(e)
    os.system("cls")
