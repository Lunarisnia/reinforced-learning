import tictactoe
import time
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Start env with random agent
env = tictactoe.TicTacToe(render_mode="console", opponent_agent=None)
# checkpoint_callback = CheckpointCallback(save_path='./tictactoe/weights', save_freq=int(1e3), name_prefix="dqn")
# model = DQN("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=int(1e4), progress_bar=True, callback=checkpoint_callback)
# model.save("./dqn_ttt")


model = DQN.load("D:\\Personal\\Workplace\\reinforced-learning\\dqn_ttt.zip")
obs, info = env.reset(player_mark=1)

actions = []

for _ in range(9):
    env.render()
    action = model.predict(obs, deterministic=True)
    actions.append(action[0])

    obs, reward, done, terminated, info = env.step(int(action[0]))
    env.render()
    print("AGENT CHOICE: ", actions)
    # time.sleep(0.5)
    # input("PRESS")
    if done:
        print(f"DONE: {info['winner']}")
        break

    if terminated:
        print("TERMINATED")
        # env.reset()
        break

env.close()
