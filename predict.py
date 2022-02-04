
import time
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from env2048 import Env2048, ACTION_MAP
from puzzle import GameGrid


def main(model_path, sleep=0.0, seed=313, render=True):

    if render:
        def matrix():
            return oenv.envs[0].env.matrix
        game_grid = GameGrid(manual=True)

    oenv = make_vec_env(Env2048, seed=seed)
    env = VecFrameStack(oenv, n_stack=2)
    model = DQN.load(model_path)

    obs = env.reset()
    step = 0
    total_reward = 0
    while True:
        if render:
            game_grid.matrix = matrix()
            game_grid.update_grid_cells()
            game_grid.update()

        action, _ = model.predict(obs, deterministic=True)
        env.render()
        obs, reward, done, _ = env.step(action)
        step += 1
        if render:
            print(f'step: {step}, action: {action[0]}, {ACTION_MAP.get(action[0])}, reward: {reward[0]}')
        total_reward += reward[0]

        if done:
            if render:
                print('total_reward:', total_reward)
                game_grid.mainloop()
            return total_reward
        else:
            if render:
                game_grid.matrix = matrix()
                game_grid.update_grid_cells()
                game_grid.update()
                if sleep > 0:
                    time.sleep(sleep)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
