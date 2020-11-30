import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers =[]
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        if j < len(sizes)-2:
            layers += [nn.Linear(sizes[j], sizes[j+1]), nn.BatchNorm1d(sizes[j+1]), act()]
        else:
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std):
        super().__init__()
        self.scan_dim = 480
        self.state_dim = 6
        self.conv1 = nn.Conv1d(1, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 64, 1, 1, 0)
        self.bn4 = nn.BatchNorm1d(64)
        self.avg_pool = nn.AvgPool1d(self.scan_dim)

        self.fc1 = nn.Linear(self.state_dim, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn6 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(128, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 2)
        self.tanh = nn.Tanh()

        self.act_mean = torch.Tensor(act_mean)
        self.act_std = torch.Tensor(act_std)

    def forward(self, obs):
        obs = obs.unsqueeze(1)
        scan, state = obs[:, :, :self.scan_dim], obs[:, :, self.scan_dim:].squeeze(1)
        scan = F.relu(self.bn1(self.conv1(scan)))
        scan = F.relu(self.bn2(self.conv2(scan)))
        scan = F.relu(self.bn3(self.conv3(scan)))
        scan = F.relu(self.bn4(self.conv4(scan)))
        scan = self.avg_pool(scan).squeeze(2)

        state = F.relu(self.bn5(self.fc1(state)))
        state = F.relu(self.bn6(self.fc2(state)))

        out = torch.cat([scan, state], dim=-1)
        out = F.relu(self.bn7(self.fc3(out)))
        out = self.tanh(self.fc4(out))

        # Return output from network scaled to action space limits.
        if out.is_cuda:
            return (self.act_std.cuda() * out) + self.act_mean.cuda()
        else:
            return (self.act_std * out) + self.act_mean


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.scan_dim = 480
        self.state_dim = 6

        self.conv1 = nn.Conv1d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv1d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv1d(128, 64, 1, 1, 0)
        self.avg_pool = nn.AvgPool1d(self.scan_dim)

        self.fc1 = nn.Linear(self.state_dim + 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, obs, act):
        obs = obs.unsqueeze(1)
        scan, state = obs[:, :, :self.scan_dim], obs[:, :, self.scan_dim:].squeeze(1)
        scan = F.relu(self.conv1(scan))
        scan = F.relu(self.conv2(scan))
        scan = F.relu(self.conv3(scan))
        scan = F.relu(self.conv4(scan))
        scan = self.avg_pool(scan).squeeze(2)

        state = torch.cat([state, act], dim=-1)
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))

        out = torch.cat([scan, state], dim=-1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return torch.squeeze(out, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        # act_limit = action_space.high[0]
        act_std = (action_space.high - action_space.low)/2
        act_mean = (action_space.high + action_space.low)/2

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
