import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from typing import NamedTuple

GAMMA = 0.99  # discount factor
TAU = 1e-3
EPSILON_START = 0.5
DECAY_RATE = 0.992
EPSILON_END = 0.01
BATCH_SIZE = 256
LR = 1e-5
BUFFER_SIZE = 500000
UPDATE_EVERY = 1000
LOSS_FUNC = nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 522


class Agent:
    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        """
        Initialize Agent
        """
        random.seed(seed)
        self.action_space = action_space
        self.observation_space = observation_space
        self.action_size = action_space.n
        self.state_size = observation_space.shape[0]
        self.qnet = QNetWork(self.state_size, self.action_size).to(device)
        self.target_net = QNetWork(self.state_size, self.action_size).to(device)
        self.replay_buffer = ReplayBuffer(self.state_size, 1, BUFFER_SIZE)
        self.lr = LR
        self.optimizer = optim.AdamW(self.qnet.parameters(), lr=LR, weight_decay=5e-4)
        self.eps = EPSILON_START
        self.total_reward = 0

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Define policy
        """
        # self.qnet.eval()
        self.eps = max(EPSILON_END, self.eps * DECAY_RATE)
        if torch.rand(1) < self.eps:
            action = np.random.randint(self.action_size)
        else:
            with torch.no_grad():
                q_pred = self.qnet(observation)
                action = np.argmax(q_pred.detach().numpy())
        # self.qnet.train()
        return action

    def learn(
        self,
        state: gym.spaces.Box,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Define learning function
        """
        if terminated or truncated:
            return
        action = self.act(observation)
        self.total_reward += reward
        self.replay_buffer.add(state, action, reward, observation, terminated)
        batch = self.replay_buffer.sample(BATCH_SIZE)
        # q_actions = self.qnet(batch.state)
        # q_pred = q_actions.gather(1, batch.action)
        q_pred = self.qnet(observation).gather(
            0, torch.tensor(action).type(torch.LongTensor)
        )
        with torch.no_grad():
            # q_next_actions = self.qnet(batch.next_state)
            # max_acts = q_next_actions.argmax(dim=1).view(-1, 1)
            # q_target_actions = self.target_net(batch.next_state)
            # q_target = q_target_actions.gather(1, max_acts)
            # q_target = batch.reward + torch.Tensor(q_target) * batch.discount
            q_target = self.qnet(observation).max(dim=0)[0].view(-1, 1)
            q_target = reward + GAMMA * q_target
        loss = LOSS_FUNC(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qnet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        soft_update_from_to(self.qnet, self.target_net)


def weight_init(w: torch.Tensor):
    """
    initialize weight
    """
    nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")
    # nn.init.normal_(w, 0, 0.1)


class QNetWork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        """
        Initialize neural net
        """
        super(QNetWork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(in_features=state_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=action_size)

        weight_init(self.fc1.weight)
        weight_init(self.fc2.weight)
        weight_init(self.fc3.weight)
        weight_init(self.fc4.weight)

    def forward(self, state):
        x = torch.tensor(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        x = self.fc3(x)
        return x


class Batch(NamedTuple):
    """
    Define namedTuple batch
    """

    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    discount: torch.Tensor
    next_state: torch.Tensor


class ReplayBuffer:
    def __init__(self, state_dim: int, act_dim: int, buffer_size: int):
        """
        Initialize replay buffer
        """
        self.buffer_size = buffer_size
        self.ptr = 0
        self.n_samples = 0

        self.state = torch.zeros(
            buffer_size, state_dim, dtype=torch.float32, device=device
        )
        self.action = torch.zeros(
            buffer_size, act_dim, dtype=torch.int64, device=device
        )
        self.reward = torch.zeros(buffer_size, 1, dtype=torch.float32, device=device)
        self.discount = torch.zeros(buffer_size, 1, dtype=torch.float32, device=device)
        self.next_state = torch.zeros(
            buffer_size, state_dim, dtype=torch.float32, device=device
        )

    def add(self, state, action, reward, next_state, terminated):
        self.state[self.ptr] = torch.Tensor(state)
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.discount[self.ptr] = GAMMA
        self.next_state[self.ptr] = torch.Tensor(next_state)

        if self.n_samples < self.buffer_size:
            self.n_samples += 1

        self.ptr = (self.ptr + 1) % UPDATE_EVERY

    def sample(self, batch_size):
        """
        sample from replay buffer
        """
        idx = np.random.choice(self.n_samples, batch_size)
        state = self.state[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        discount = self.discount[idx]
        next_state = self.next_state[idx]

        return Batch(state, action, reward, discount, next_state)


def soft_update_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
