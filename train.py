
from env2048 import Env2048
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from features import CustomCNN, CustomCNNBN, CustomCNNRes, CustomCNN128, CustomCNNLSTM

feature_mapping = {
    'CustomCNN': CustomCNN,
    'CustomCNN128': CustomCNN128,
    'CustomCNNBN': CustomCNNBN,
    'CustomCNNRes': CustomCNNRes,
    'CustomCNNLSTM': CustomCNNLSTM
}


def train(
    n_stack=2,
    feature='CustomCNN',
    epochs=100,
    n_envs=12,
    train_steps=10_0000,
    n_test=50,
    save_dir='models',
    gamma=0.9,
    learning_rate=1e-3
):
    check_env(Env2048())
    assert feature in feature_mapping
    test_env = VecFrameStack(make_vec_env(Env2048), n_stack=n_stack)

    model = DQN(
        'CnnPolicy',
        VecFrameStack(make_vec_env(Env2048, n_envs), n_stack=n_stack),
        buffer_size=20_0000,
        gamma=gamma,
        learning_rate=learning_rate,
        batch_size=128,
        verbose=1,
        policy_kwargs = dict(
            features_extractor_class=feature_mapping[feature],
            features_extractor_kwargs={},
        )
    )

    steps = 0
    for e in range(epochs):
        model.learn(total_timesteps=train_steps, reset_num_timesteps=False)
        steps += train_steps
        obs = test_env.reset()
        total_rewards = 0
        for _ in range(n_test):
            total_reward = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                test_env.render()
                total_reward += reward[0]
                if done:
                    obs = test_env.reset()
                    print(total_reward)
                    total_rewards += total_reward
                    break
        total_rewards //= n_test
        model.save(f'{save_dir}/{e + 1}_{steps}_{int(total_rewards)}')


def main():
    import os
    from itertools import product
    from tqdm import tqdm
    feature = 'CustomCNN', 'CustomCNN128', 'CustomCNNRes', 'CustomCNNBN', 'CustomCNNLSTM'
    n_stack = 2,
    gamma = 0.5, 0.7, 0.9, 0.99, 0.999
    learning_rate = 1e-3, 1e-4,
    params = [
        {
            'feature': feature,
            'n_stack': n_stack,
            'gamma': gamma,
            'learning_rate': learning_rate
        }
        for feature, n_stack, gamma, learning_rate in product(feature, n_stack, gamma, learning_rate)
    ]
    print(len(params))
    for p in tqdm(params):
        save_dir = f'models_{p["feature"]}_{p["n_stack"]}_{p["gamma"]}_{p["learning_rate"]}'
        if os.path.exists(save_dir):
            continue
        p['save_dir'] = save_dir
        train(**p)


if __name__ == '__main__':
    # main()
    # train()
    from fire import Fire
    Fire(train)
