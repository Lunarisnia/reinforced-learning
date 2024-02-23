import gymnasium
from gymnasium import spaces
from game import LinSnake


class LinSnakeEnv(gymnasium.Env):
    linSnake = None
    metadata = {'render_modes': ["console"]}

    def __init__(self, render_mode=None, road_length=16, goal=32):
        self.render_mode = render_mode
        self.road_length = road_length
        self.goal = goal

        self.action_space = spaces.Discrete(2)
        # TODO: Make linsnake return array of the obs space if its done this way
        self.observation_space = spaces.Box(low=0, high=self.road_length - 1, shape=(road_length,))

        self.linSnake = LinSnake(road_length=self.road_length, goal=self.goal)

    def step(self, action):
        self.linSnake.move_snake(action)
        # TODO: Calculate obs, reward, etc

    def reset(self):
        self.linSnake.reset()
        # TODO: Return observation

    def render(self):
        if self.render_mode != "console":
            raise NotImplementedError("render mode not implemented")
        self.linSnake.render()

    def close(self):
        pass
