import gymnasium
import numpy as np
from gymnasium import spaces
from game import LinSnake, GameState


class LinSnakeEnv(gymnasium.Env):
    linSnake = None
    metadata = {'render_modes': ["console"]}

    def __init__(self, render_mode=None, road_length=16, goal=32):
        self.render_mode = render_mode
        self.road_length = road_length
        self.goal = goal

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(road_length,))
        self.linSnake = LinSnake(road_length=self.road_length, goal=self.goal)

    def step(self, action):
        self.linSnake.move_snake(action)
        self.linSnake.calculate_state()
        done = False
        if self.linSnake.state == GameState.WON:
            done = True
        terminated = False
        if self.linSnake.state == GameState.LOST:
            terminated = True

        obs = self._make_observation()
        reward = self._calculate_reward()
        return obs, reward, done, terminated, {}

    def _calculate_reward(self):
        if self.linSnake.state == GameState.WON:
            return 1e6
        if self.linSnake.state == GameState.LOST:
            return -1e3
        return -0.01

    def _make_observation(self):
        obs = np.zeros(shape=(self.linSnake.road_length,))
        obs[self.linSnake.player_pos] = 1.
        obs[self.linSnake.food_location] = -1.
        return obs

    def reset(self, **kwargs):
        self.linSnake.reset()
        obs = self._make_observation()
        return obs, {}

    def render(self):
        if self.render_mode != "console":
            raise NotImplementedError("render mode not implemented")
        self.linSnake.render()

    def close(self):
        pass

# env = LinSnakeEnv(render_mode="console")
# obs, info = env.reset()
#
# while True:
#     env.render()
#     action = int(input("Action: "))
#
#     obs, reward, done, terminated, info = env.step(action)
#
#     if done:
#         print("Won")
#         break
#     if terminated:
#         print("Lost")
#         break
#
# env.close()