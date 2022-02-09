

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch as th


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


class CustomCNNLSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super(CustomCNNLSTM, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = 1024  # self.cnn2(self.cnn(th.as_tensor(observation_space.sample()[None]).float())).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(features_dim, features_dim, 1, batch_first=True)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.size()[0]
        seq = th.stack(th.split(observations, 4, dim=2))
        seq = seq.view(batch_size * 2, 16, 4, 4)
        seq = self.linear(self.cnn2(self.cnn(seq)))
        seq = seq.view(batch_size, 2, -1)
        hidden_state, _ = self.lstm(seq)
        last_hidden_state = hidden_state[:, -1, :]
        return last_hidden_state


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

