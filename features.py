

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch as th


class ResBlock(nn.Module):
    def __init__(self, dim=64):
        super(ResBlock, self).__init__()
        self.cnn1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)
    
    def forward(self, x):
        t = self.relu(self.bn1(self.cnn1(x)))
        t = self.bn2(self.cnn2(t))
        x = x + t
        x = self.relu(x)
        return x


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn2(self.cnn(th.as_tensor(observation_space.sample()[None]).float())).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn2(self.cnn(observations)))


class CustomCNN128(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super(CustomCNN128, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn2(self.cnn(th.as_tensor(observation_space.sample()[None]).float())).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn2(self.cnn(observations)))


class CustomCNNBN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super(CustomCNNBN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn2(self.cnn(th.as_tensor(observation_space.sample()[None]).float())).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn2(self.cnn(observations)))


class CustomCNNRes(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super(CustomCNNRes, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.cnn2 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn2(self.cnn(th.as_tensor(observation_space.sample()[None]).float())).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn2(self.cnn(observations)))

