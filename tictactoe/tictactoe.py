# Import required libraries
import os
import random
import numpy as np
from gymnasium import spaces
from gymnasium import Env

# There is a rare bug where random agent chose a grid that already occupied

class RandomAgent:
    """
    This class represents a simple random agent. It randomly selects a move to play on the board.
    :param mark: The mark (1 or 2) that this agent will use to make its moves on the board.
    """

    def __init__(self, mark):
        self.mark = mark

    def play_move(self, board):
        # Make a copy of the board to avoid modifying the original one
        next_board = board.copy()
        valid_moves = [(row, col) for row in range(len(next_board)) for col in range(len(next_board[row])) if
                       next_board[row][col] == 0]
        # Select a random column and row for the move
        if len(valid_moves) == 0:
            return next_board
        col, row = random.choice(valid_moves)
        # Make the move by placing the agent's mark on the selected position
        next_board[row][col] = self.mark
        return next_board


class TicTacToe(Env):
    """
    This class represents a simple Tic-Tac-Toe game environment. It uses the OpenAI Gym framework for its structure and functionality.
    :param render_mode: The mode in which to render the game. Supported modes are "human" (show the game visually) and "console" (print the game state to the console).
    :param opponent_agent: A reference to the agent that will play against this environment's agent. If not provided, the game will be played with a simple random agent instead.
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, render_mode=None, opponent_agent=None):
        self.board = np.zeros((3, 3), dtype=np.float32)
        self.render_mode = render_mode
        self.current_player = 1

        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 3), dtype=np.float32)
        # The game can be in one of 9 states (representing the possible moves on a tic-tac-toe board).
        self.action_space = spaces.Discrete(9)

        self.action_to_coordinate = {
            0: (0, 0),
            1: (0, 1),
            2: (0, 2),
            3: (1, 0),
            4: (1, 1),
            5: (1, 2),
            6: (2, 0),
            7: (2, 1),
            8: (2, 2),
        }
        # If an opponent agent is provided, it will be used to play the game.
        # Otherwise, a simple random agent will be used.
        self.opponent_agent = opponent_agent if opponent_agent else RandomAgent
        # Randomly assign the player mark (either 1 or 2) to each player.
        self._randomize_player_mark()

    def _randomize_player_mark(self, player_mark=None):
        self.player_mark = player_mark if player_mark else np.random.randint(1, 3)
        self.opponent_mark = 1 if self.player_mark == 2 else 2
        # Create a dictionary to hold the players and their marks.
        self.players = {
            self.player_mark: "self",
            self.opponent_mark: self.opponent_agent(self.opponent_mark)
        }

    def _turn_keeper(self, move):
        """
        This function is used to keep track of whose turn it is to make a move. It ensures that each player gets their chance to play.
        :param move: The move made by the current player.
        """
        if self.player_mark == self.current_player:
            # If it's the player's turn, update the board and switch to the opponent's turn.
            self.board = self._make_move(move)
            self.current_player = self.opponent_mark
            if not self._is_terminal()[0]:
                self._turn_keeper(move)
        else:
            # If it's the opponent's turn, let the opponent make a move and then switch back to the player's turn.
            self.board = self.players[self.opponent_mark].play_move(self.board)
            self.current_player = self.player_mark

    def reset(self, player_mark = None, **kwargs):
        """
        This function is used to reset the game environment to its initial state.
        :param kwargs: Additional keyword arguments that can be passed to this function (not used in this implementation).
        :return: The current state of the board and an empty dictionary for any additional information.
        """
        self._randomize_player_mark(player_mark)
        self.board = np.zeros((3, 3), dtype=np.float32)
        self.current_player = 1
        # If it's the opponent's turn to make the first move,
        # let the opponent play and then switch to the player's turn.
        if self.opponent_mark == self.current_player:
            self.board = self.players[self.opponent_mark].play_move(self.board)
            self.current_player = self.player_mark
        return self.board, {}

    def step(self, action):
        """
        This function is used to execute a single move in the game environment and update the state of the game accordingly.
        :param action: The index of the cell on the board where the player wants to make their move (ranges from 0 to 8).
        :return: The new state of the board, a reward value of 0 (since we are not yet calculating rewards), a boolean indicating whether the game has ended due to a win or draw, another boolean indicating whether the game has ended because it is a loss for the current player, and an empty dictionary for any additional information.
        """
        terminated = False
        done = False
        try:
            self._turn_keeper(action)
        except ValueError:
            terminated = True
            reward = -1e8
            return self.board, reward, done, terminated, {}
        reward = self._calculate_reward()
        terminal, winner = self._is_terminal()
        if terminal:
            done = True

        return self.board, reward, done, terminated, {
            "winner": winner,
        }

    def _make_move(self, move):
        """
        This function is used to update the state of the board after a move has been made by a player.
        :param move: The index of the cell on the board where the player wants to make their move (ranges from 0 to 8).
        :return: The updated state of the board.
        """
        row, col = self.action_to_coordinate[move]
        # If the selected cell is already occupied, raise an error.
        if self.board[row][col] != 0:
            raise ValueError(f"Grid is occupied")
        next_board = self.board.copy()
        next_board[row][col] = self.player_mark
        return next_board

    def _calculate_reward(self):
        terminal_reward_dict = {
            self.player_mark: int(1e5),
            self.opponent_mark: int(-1e3),
            0: 0,
        }

        is_terminal, winner = self._is_terminal()
        if is_terminal:
            return terminal_reward_dict[winner]

        return 1e-1

    def _is_terminal(self):
        board = self.board.copy()

        # Check Horizontal Axes
        for row in range(len(board)):
            if np.all(board[row] == self.player_mark):
                return True, self.player_mark
            elif np.all(board[row] == self.opponent_mark):
                return True, self.opponent_mark

        # Check Vertical Axes
        for row in range(len(board)):
            if np.all(board[:, row:row + 1] == self.player_mark):
                return True, self.player_mark
            elif np.all(board[:, row:row + 1] == self.opponent_mark):
                return True, self.opponent_mark

        # Check Positive Diagonal
        if np.all(board.diagonal() == self.player_mark):
            return True, self.player_mark
        elif np.all(board.diagonal() == self.opponent_mark):
            return True, self.opponent_mark

        # Check Negative Diagonal
        if np.all(board[::-1].diagonal() == self.player_mark):
            return True, self.player_mark
        elif np.all(board[::-1].diagonal() == self.opponent_mark):
            return True, self.opponent_mark

        # Check draw
        if not np.any(board == 0):
            return True, 0

        return False, 0

    # Render the game
    def render(self):
        if self.render_mode != "console":
            raise NotImplementedError()
        os.system("cls")
        for row in self.board:
            for col in row:
                print(" X " if col == 1 else " O " if col == 2 else " * ", sep="", end="")
            print()
        print(f"Agent: {self.player_mark}")
        print(f"Opponent: {self.opponent_mark}")

    # TODO: Close anything related to the env
    def close(self):
        pass


# env = TicTacToe(render_mode="console", opponent_agent=RandomAgent)
# obs, info = env.reset()
#
# for _ in range(9):
#     # env.render()
#     action = int(input("Input your move: "))
#
#     obs, reward, done, terminated, info = env.step(action)
#     # env.render()
#     if done:
#         print("DONE")
#         break
#
#     if terminated:
#         print("TERMINATED")
#         # env.reset()
#         break
#
#
# env.close()