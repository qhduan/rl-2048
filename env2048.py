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

    def __init__(self, n=4, max_idle=100, flatten=False, seed=None):
        super(Env2048, self).__init__()
        self.flatten = flatten
        self.n = n
        self.max_idle = max_idle
        self.action_map = ACTION_MAP
        # up, down, left, right
        self.action_space = gym.spaces.Discrete(4)
        if self.flatten:
            self.observation_space = gym.spaces.Box(0.0, 2048.0, [self.n * self.n,])
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(self.n, self.n, 1025), dtype=np.uint8)
            self.eye = np.eye(1025)
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
        if self.flatten:
            obs = np.array(self.matrix).flatten()
        else:
            m = np.array(self.matrix)
            m = m // 2  # from 2,4,6,...,2048 to 0,1,2,...,1023
            # m = nn.functional.one_hot(torch.LongTensor(m), 1025)
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
        reward = np.sum((new_score - old_score) * (new_score >= old_score)) * 2
        reward = float(reward)
        self.total_reward += reward

        self.i += 1
        if updated:  # matrix有更新
            self.matrix = logic.add_two(self.matrix)

            if logic.game_state(self.matrix) == 'win':
                # 胜利
                return self.obs, 100000.0, True, {'i': self.i, 'ri': self.reward_i, 'tr': self.total_reward}
            elif logic.game_state(self.matrix) == 'lose':
                # 失败
                return self.obs, 1000.0, True, {'i': self.i, 'ri': self.reward_i, 'tr': self.total_reward}

        # 要求矩阵至少变换，避免无效按键出现
        if old_matrix == new_matrix:
            reward -= 10

        if reward > 0:
            self.reward_i = self.i
        elif self.i - self.reward_i > self.max_idle:
            # 如果太长时间没变化，就结束
            return self.obs, -5000, True, {'i': self.i, 'ri': self.reward_i, 'tr': self.total_reward}

        return self.obs, reward, False, {'i': self.i, 'ri': self.reward_i, 'tr': self.total_reward}

    def render(self, mode='human'):
        # print(np.array(self.matrix))
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
