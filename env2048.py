import logic
import numpy as np
import gym


ACTION_MAP = {
    0: 'up',
    1: 'down',
    2: 'left',
    3: 'right'
}


class Env2048(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, n=4, max_idle=100, seed=None):
        super(Env2048, self).__init__()
        self.n = n
        self.max_idle = max_idle
        self.action_map = ACTION_MAP
        # up, down, left, right
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self.n, self.n, 2 ** n), dtype=np.uint8)
        self.eye = np.eye(2 ** n)
        self.reward_range = (float('-inf'), float('inf'))
        if seed is not None:
            self.seed(seed)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.matrix = logic.new_game(self.n)
        self.reward_i = self.i = 0
        self.total_reward = 0
        return self.obs

    @property
    def obs(self):
        m = np.array(self.matrix)
        m = np.clip(m, 1, float('inf'))  # from 0, 2, 4, 8, ... to 1, 2, 4, 8
        m = np.log2(m).astype(np.int64)  # from 1, 2, 4, 8,..., 2048 to 0, 1, 2, 3, ..., 11
        m = self.eye[m]
        m = m * 255
        m = m.astype(np.uint8)
        obs = m
        return obs

    def step(self, action):

        if isinstance(action, str) and action in ('up', 'down', 'left', 'right'):
            pass
        if isinstance(action, (int, np.int64, np.int32)):
            action = self.action_map[int(action)]
        else:
            print(action, type(action))
            raise
    
        old_score = np.sort(np.array(self.matrix).flatten())[::-1]
        old_matrix = str(self.matrix)
        # import pdb; pdb.set_trace()
        if action == 'up':
            self.matrix, updated = logic.up(self.matrix)
        elif action == 'down':
            self.matrix, updated = logic.down(self.matrix)
        elif action == 'left':
            self.matrix, updated = logic.left(self.matrix)
        elif action == 'right':
            self.matrix, updated = logic.right(self.matrix)

        new_matrix = str(self.matrix)
        new_score = np.sort(np.array(self.matrix).flatten())[::-1]
        reward = np.sum((new_score - old_score) * (new_score >= old_score)) * 4
        reward = float(reward)
        self.total_reward += reward

        self.i += 1
        if updated:  # matrix有更新
            self.matrix = logic.add_two(self.matrix)

            if logic.game_state(self.matrix) == 'win':
                print('you win')
                return self.obs, 10000.0, True, {'i': self.i, 'ri': self.reward_i, 'tr': self.total_reward}
            elif logic.game_state(self.matrix) == 'lose':
                return self.obs, 100.0, True, {'i': self.i, 'ri': self.reward_i, 'tr': self.total_reward}

        idle = False
        if old_matrix == new_matrix:
            idle = True

        if idle:
            reward = -1
        else:
            self.reward_i = self.i

        if self.i - self.reward_i > self.max_idle:
            return self.obs, -100, True, {'i': self.i, 'ri': self.reward_i, 'tr': self.total_reward}

        return self.obs, reward, False, {'i': self.i, 'ri': self.reward_i, 'tr': self.total_reward}

    def render(self, mode='human'):
        pass
    
    def close(self):
        pass


def main():
    env = Env2048()
    obs = env.reset()
    print(obs)
    for _ in range(1000):
        obs, reward, done, info = env.step(np.random.choice(['right', 'left', 'up', 'down']))
        print(obs)
        print(reward, done, info)
        if done:
            break


if __name__ == '__main__':
    main()
