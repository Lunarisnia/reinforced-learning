import time

import gymnasium
import numpy as np
from enum import Enum
from gymnasium import spaces


class Actions(Enum):
    RIGHT = 0
    LEFT = 1


class GoLeftEnv(gymnasium.Env):
    metadata = {"render_modes": ["console"], "step_limit": 0, "current_step": 0, "current_reward": 0}

    def __init__(self, render_mode=None, road_length=16):
        self.render_mode = render_mode
        self.road_length = road_length
        self.agent_position = self.road_length - 1

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=self.road_length, shape=(1,), dtype=np.float32)

        self.metadata["step_limit"] = self.road_length + 10

    def reset(self, **kwargs):
        self.metadata["current_reward"] = 0
        self.metadata["current_step"] = 0
        self.agent_position = self.road_length - 1
        return np.array([self.agent_position]).astype(np.float32), {}

    def step(self, action):
        if action == Actions.LEFT.value:
            self.agent_position -= 1
        elif action == Actions.RIGHT.value:
            self.agent_position += 1
        else:
            raise ValueError(f"Received invalid action: {action}")

        self.agent_position = np.clip(self.agent_position, 0, self.road_length)

        terminated = self.metadata["current_step"] > self.metadata["step_limit"]

        done = self.agent_position == 0

        reward = self._calculate_reward(action)
        self.metadata["current_reward"] += reward

        info = {}

        self.metadata["current_step"] += 1
        return np.array([self.agent_position]), reward, done, terminated, info

    def _calculate_reward(self, action):
        if self.agent_position == 0:
            return 1e5
        elif action == Actions.LEFT.value:
            return self.metadata["current_step"] * self.road_length * 3e-2
        else:
            return self.metadata["current_step"] * self.road_length * (-3e-5)

    def render(self):
        if self.render_mode != "console":
            raise NotImplementedError()
        # Agent is a + the rest is a dot
        print("." * self.agent_position, end="")
        print("+", end="")
        print("." * (self.road_length - self.agent_position))

    def close(self):
        pass


# Best agent hardcoded
# import time
# import os
# env = GoLeftEnv(render_mode="console")
#
# obs, info = env.reset()
#
# GO_LEFT = Actions.LEFT
# n_steps = 17
# for step in range(n_steps):
#     obs, reward, done, terminated, info = env.step(GO_LEFT)
#
#     time.sleep(1)
#     os.system("cls")
#     env.render()
#
#     if done or terminated:
#         env.reset()
#
# env.close()

import os
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

env = GoLeftEnv(render_mode="console")
checkpoint_callback = CheckpointCallback(save_freq=int(1e3), save_path="./flappyBirds/", name_prefix="dqn")
model = DQN("MlpPolicy", env, verbose=1)
# 100 Episode here is best because it wont overfit too bad
# The previous was 5000 which result in a model that can only navigate through road that is 16 in length
model.learn(total_timesteps=100, progress_bar=True, callback=checkpoint_callback)
model.save("./dqn_goleft")

# env = GoLeftEnv(render_mode="console")
# env = make_vec_env(lambda: env, n_envs=1)
# checkpoint_callback = CheckpointCallback(save_freq=int(1e3), save_path="./flappyBirds/", name_prefix="ppo")
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=5000, progress_bar=True, callback=checkpoint_callback)
# model.save("./ppo_goleft")


# model = PPO.load("./ppo_goleft.zip")
#
# env = GoLeftEnv(render_mode="console")
# env = make_vec_env(lambda: env, n_envs=1)
#
# obs = env.reset()
#
# n_steps = 10000
# for step in range(n_steps):
#     action = model.predict(obs)
#     obs, reward, done, terminated = env.step(action[0])
#
#     time.sleep(0.2)
#     os.system("cls")
#     env.render()
#     print(f"current_reward: {env.metadata['current_reward']}")
#
# env.close()
