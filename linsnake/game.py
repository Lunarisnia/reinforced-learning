from enum import Enum
import numpy as np

class GameState(Enum):
    PLAYING = "PLAYING"
    WON = "WON"
    LOST = "LOST"

class LinSnake:
    counter = 0
    goal = 0
    max_step = 0
    step = 0

    player = " O "
    road = " . "
    food = " * "

    left = 0
    right = 1

    state = GameState.PLAYING

    def __init__(self, road_length=16, goal=32):
        if road_length < 3:
            raise ValueError("Road is too short")
        if goal < 1:
            raise ValueError("Goal cannot be less than 1")
        self.road_length = road_length
        self.player_pos = np.clip((self.road_length - 1) // 2, 0, road_length - 1)
        self.food_location = 0
        self.goal = goal
        self.max_step = road_length

    def spread_food(self):
        self.food_location = np.random.randint(0, self.road_length)
        if self.food_location == self.player_pos:
            self.spread_food()

    def move_snake(self, action):
        if action == 0:
            self.player_pos -= 1
        elif action == 1:
            self.player_pos += 1
        else:
            raise ValueError("Invalid action")

        self.step += 1
        self.player_pos = np.clip(self.player_pos, 0, self.road_length - 1)

    def check_ate(self):
        if self.food_location == self.player_pos:
            return True
        return False

    def count_score(self):
        if self.check_ate():
            self.counter += 1

    def calculate_state(self):
        self.count_score()
        if self.step > self.max_step:
            self.state = GameState.LOST
        if self.counter >= self.goal:
            self.state = GameState.WON
        if self.check_ate() and self.state == GameState.PLAYING:
            self.step = 0
            self.spread_food()

    def render(self):
        for x in range(self.road_length):
            if self.player_pos == x:
                print(self.player, end='')
            elif self.food_location == x:
                print(self.food, end='')
            else:
                print(self.road, end='')
        print("")

    def reset(self):
        self.player_pos = np.clip((self.road_length - 1) // 2, 0, self.road_length - 1)
        self.food_location = 0
        self.step = 0


# game = LinSnake(goal=3)
# game.spread_food()
#
# while GameState.PLAYING == game.state:
#     game.render()
#     action = int(input("action: "))
#     game.move_snake(action)
#     game.calculate_state()
#
# game.render()
# result = "You " + game.state.value
# print(result)