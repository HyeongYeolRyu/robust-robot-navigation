import numpy as np
# import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std):
        super().__init__()
        self.scan_dim = 480
        self.state_dim = 6
        self.fc1 = nn.Linear(self.scan_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(self.state_dim, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)

        self.fc6 = nn.Linear(64, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.lstm = nn.LSTMCell(input_size=64, hidden_size=64)
        self.reset_lstm()

        self.fc7 = nn.Linear(64, 2)
        # self.bn7 = nn.BatchNorm1d(2) # TODO
        self.tanh = nn.Tanh()

        # pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        # self.pi = mlp(pi_sizes, activation, output_activation=nn.Tanh)
        self.act_mean = torch.Tensor(act_mean)
        self.act_std = torch.Tensor(act_std)

    def forward(self, obs):
        scan, state = obs[:, :self.scan_dim], obs[:, self.scan_dim:]
        # print(f'state:{state}')

        scan = F.relu(self.bn1(self.fc1(scan)))
        scan = F.relu(self.bn2(self.fc2(scan)))
        scan = F.relu(self.bn3(self.fc3(scan)))

        state = F.relu(self.bn4(self.fc4(state)))
        state = F.relu(self.bn5(self.fc5(state)))

        out = torch.cat([scan, state], dim=-1)
        out = F.relu(self.bn6(self.fc6(out)))

        self.hx = self.hx.to(out.device)
        self.cx = self.cx.to(out.device)
        if self.reset_flag:
            batch_size = out.size(0)
            self.hx = self.hx.repeat(batch_size, 1)
            self.cx = self.cx.repeat(batch_size, 1)
        if out.size(0) == 1:
            self.hx = torch.zeros(1, 64, device=out.device)
            self.cx = torch.zeros(1, 64, device=out.device)
        self.hx, self.cx = self.lstm(out, (self.hx, self.cx))
        self.reset_flag = False
        out = self.hx

        out = self.tanh(self.fc7(out))
        # print(f"out {out}")

        # Return output from network scaled to action space limits.
        if out.is_cuda:
            return (self.act_std.cuda() * out) + self.act_mean.cuda()
        else:
            return (self.act_std * out) + self.act_mean

    def reset_lstm(self):
        self.hx = torch.zeros(1, 64)
        self.cx = torch.zeros(1, 64)
        self.reset_flag = True

    # def forward(self, obs):
    #     out = self.pi(obs)
    #     # print(f"out {out}")
    #     # Return output from network scaled to action space limits.
    #     if obs.is_cuda:
    #         return (self.act_std.cuda() * out) + self.act_mean.cuda()
    #     else:
    #         return (self.act_std * out) + self.act_mean


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.scan_dim = 480
        self.state_dim = 6
        self.fc1 = nn.Linear(self.scan_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(self.state_dim+2, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)

        self.fc6 = nn.Linear(64, 64)
        # self.bn6 = nn.BatchNorm1d(64)
        self.lstm = nn.LSTMCell(input_size=64, hidden_size=64)
        self.reset_lstm()

        self.fc7 = nn.Linear(64, 1)

    def forward(self, obs, act):
        scan, state = obs[:, :self.scan_dim], obs[:, self.scan_dim:]
        scan = F.relu(self.fc1(scan))
        scan = F.relu(self.fc2(scan))
        scan = F.relu(self.fc3(scan))

        state = torch.cat([state, act], dim=-1)
        state = F.relu(self.fc4(state))
        state = F.relu(self.fc5(state))

        out = torch.cat([scan, state], dim=-1)
        out = F.relu(self.fc6(out))
        self.hx = self.hx.to(out.device)
        self.cx = self.cx.to(out.device)
        if self.reset_flag:
            batch_size = out.size(0)
            self.hx = self.hx.repeat(batch_size, 1)
            self.cx = self.cx.repeat(batch_size, 1)
        if out.size(0) == 1:
            self.hx = torch.zeros(1, 64, device=out.device)
            self.cx = torch.zeros(1, 64, device=out.device)
        self.hx, self.cx = self.lstm(out, (self.hx, self.cx))
        self.reset_flag = False
        out = self.hx

        out = self.fc7(out)

        return torch.squeeze(out, -1)  # Critical to ensure q has right shape.

    def reset_lstm(self):
        self.hx = torch.zeros(1, 64)
        self.cx = torch.zeros(1, 64)
        self.reset_flag = True


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


    def reset_lstm(self):
        self.pi.reset_lstm()
        self.q.reset_lstm()