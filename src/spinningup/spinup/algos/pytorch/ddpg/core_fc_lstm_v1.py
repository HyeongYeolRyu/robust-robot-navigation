import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


sensor_dim = 480
print_verbose = False
hidden_features = np.array([512, 512, 64]).tolist()
use_lstm = True
lstm_hidden_state = hidden_features[-1] * 2
actor_activation = 'relu'
critic_activation = 'relu'
actor_bn = True
critic_bn = False


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        if j < len(sizes) - 2:
            layers += [nn.Linear(sizes[j], sizes[j + 1]), nn.BatchNorm1d(sizes[j + 1]), act()]
        else:
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class FcBlock(nn.Module):

    def __init__(self, bn=True, activation='relu', **kwargs):
        super(FcBlock, self).__init__()
        self.fc = nn.Linear(**kwargs)
        if bn:
            self.batchnorm = nn.BatchNorm1d(self.fc.out_features)
        if activation == 'relu':
            self.activation = nn.ReLU(False if use_lstm else True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=False if use_lstm else True)

        self.bn = bn

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.batchnorm(x)
        x = self.activation(x)
        if print_verbose:
            print(x.size())

        return x


class SensorEmbedding(nn.Module):
    def __init__(self, bn, activation):
        super(SensorEmbedding, self).__init__()

        self.fc_layers = nn.Sequential(
            FcBlock(
                bn=bn,
                activation=activation,
                in_features=sensor_dim,
                out_features=hidden_features[0]
            ),
            FcBlock(
                bn=bn,
                activation=activation,
                in_features=hidden_features[0],
                out_features=hidden_features[1],
            ),
            FcBlock(
                bn=bn,
                activation=activation,
                in_features=hidden_features[1],
                out_features=hidden_features[2],
            )
        )

    def forward(self, scan):
        scan = self.fc_layers(scan)

        return scan


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std):
        super().__init__()
        self.scan_dim = 480
        self.state_dim = 6

        # scan
        self.scan_embeddinglayers = SensorEmbedding(bn=actor_bn, activation=actor_activation)

        # state
        self.fc1 = nn.Linear(self.state_dim, hidden_features[-1] // 2)
        self.bn1 = nn.BatchNorm1d(hidden_features[-1] // 2)
        self.fc2 = nn.Linear(hidden_features[-1] // 2, hidden_features[-1])
        self.bn2 = nn.BatchNorm1d(hidden_features[-1])

        # scan & state
        self.fc3 = nn.Linear(hidden_features[-1] * 2, hidden_features[-1] * 2)
        self.bn3 = nn.BatchNorm1d(hidden_features[-1] * 2)
        if use_lstm:
            self.lstm = nn.LSTMCell(input_size=hidden_features[-1] * 2, hidden_size=lstm_hidden_state)
            self.reset_lstm()

        self.fc4 = nn.Linear(hidden_features[-1] * 2, 2)
        self.tanh = nn.Tanh()

        self.act_mean = torch.Tensor(act_mean)
        self.act_std = torch.Tensor(act_std)

    def forward(self, obs):  # obs : (B, sensor_dim+6)

        scan, state = obs[:, :self.scan_dim], obs[:, self.scan_dim:]

        # scan
        scan = self.scan_embeddinglayers(scan)

        # state
        state = F.relu(self.bn1(self.fc1(state)))
        state = F.relu(self.bn2(self.fc2(state)))

        # scan & state
        out = torch.cat([scan, state], dim=-1)
        out = F.relu(self.bn3(self.fc3(out)))

        if use_lstm:
            self.hx = self.hx.to(out.device)
            self.cx = self.cx.to(out.device)
            if self.reset_flag:
                batch_size = out.size(0)
                self.hx = self.hx.repeat(batch_size, 1)
                self.cx = self.cx.repeat(batch_size, 1)
            if out.size(0) == 1:
                self.hx = torch.zeros(1, lstm_hidden_state, device=out.device)
                self.cx = torch.zeros(1, lstm_hidden_state, device=out.device)
            self.hx, self.cx = self.lstm(out, (self.hx, self.cx))
            self.reset_flag = False
            out = self.hx

        out = self.fc4(out)
        out = self.tanh(out)

        # Return output from network scaled to action space limits.
        if out.is_cuda:
            return (self.act_std.cuda() * out) + self.act_mean.cuda()
        else:
            return (self.act_std * out) + self.act_mean

    def reset_lstm(self):
        self.hx = torch.zeros(1, lstm_hidden_state)
        self.cx = torch.zeros(1, lstm_hidden_state)
        self.reset_flag = True


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.scan_dim = 480
        self.state_dim = 6

        # scan
        self.scan_embeddinglayers = SensorEmbedding(bn=critic_bn, activation=critic_activation)

        # act & state
        self.fc1 = nn.Linear(self.state_dim + 2, hidden_features[-1] // 2)
        self.fc2 = nn.Linear(hidden_features[-1] // 2, hidden_features[-1])

        # scan & state+act
        self.fc3 = nn.Linear(hidden_features[-1] * 2, hidden_features[-1] * 2)
        if use_lstm:
            self.lstm = nn.LSTMCell(input_size=hidden_features[-1] * 2, hidden_size=lstm_hidden_state)
            self.reset_lstm()
        self.fc4 = nn.Linear(hidden_features[-1] * 2, 1)

    def forward(self, obs, act):
        scan, state = obs[:, :self.scan_dim], obs[:, self.scan_dim:]

        # scan
        scan = self.scan_embeddinglayers(scan)

        # act & state
        state = torch.cat([state, act], dim=-1)
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))

        # scan & state+act
        out = torch.cat([scan, state], dim=-1)
        out = F.relu(self.fc3(out))
        if use_lstm:
            self.hx = self.hx.to(out.device)
            self.cx = self.cx.to(out.device)
            if self.reset_flag:
                batch_size = out.size(0)
                self.hx = self.hx.repeat(batch_size, 1)
                self.cx = self.cx.repeat(batch_size, 1)
            if out.size(0) == 1:
                self.hx = torch.zeros(1, lstm_hidden_state, device=out.device)
                self.cx = torch.zeros(1, lstm_hidden_state, device=out.device)
            self.hx, self.cx = self.lstm(out, (self.hx, self.cx))
            self.reset_flag = False
            out = self.hx

        out = self.fc4(out)

        return torch.squeeze(out, -1)  # Critical to ensure q has right shape.

    def reset_lstm(self):
        self.hx = torch.zeros(1, lstm_hidden_state)
        self.cx = torch.zeros(1, lstm_hidden_state)
        self.reset_flag = True


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        # act_limit = action_space.high[0]
        act_std = (action_space.high - action_space.low) / 2
        act_mean = (action_space.high + action_space.low) / 2

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

    def reset_lstm(self):
        if use_lstm:
            self.pi.reset_lstm()
            self.q.reset_lstm()

        else:
            pass



