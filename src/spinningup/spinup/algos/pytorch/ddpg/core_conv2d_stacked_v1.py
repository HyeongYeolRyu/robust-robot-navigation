import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print_verbose = False
hidden_channels = np.array([32, 64, 64, 128, 128, 256], dtype=np.int).tolist()
actor_activation = 'relu'
critic_activation = 'relu'
actor_bn = True
critic_bn = False

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


class ConvBlock(nn.Module):

    def __init__(self, bn=True, activation='relu', **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(**kwargs, bias=False if bn else True)
        if bn:
            self.batchnorm = nn.BatchNorm2d(self.conv.out_channels)
        if activation == 'relu':
            self.activation = nn.ReLU(True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)

        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.batchnorm(x)
        x = self.activation(x)
        if print_verbose:
            print(x.size())

        return x


class SensorEmbedding(nn.Module):
    def __init__(self, bn, activation):

        super(SensorEmbedding, self).__init__()
        self.conv_layers = nn.Sequential(
            ConvBlock(
                bn=bn,
                activation=activation,
                in_channels=1,
                out_channels=hidden_channels[0],
                kernel_size=(3, 3),
                stride=(1, 2),
                padding=(1, 1)
            ),  # 30 240
            ConvBlock(
                bn=bn,
                activation=activation,
                in_channels=hidden_channels[0],
                out_channels=hidden_channels[1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),  # 15 120
            ConvBlock(
                bn=bn,
                activation=activation,
                in_channels=hidden_channels[1],
                out_channels=hidden_channels[2],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            ),  # 8 60
            ConvBlock(
                bn=bn,
                activation=activation,
                in_channels=hidden_channels[2],
                out_channels=hidden_channels[3],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            ),  # 4 30
            ConvBlock(
                bn=bn,
                activation=activation,
                in_channels=hidden_channels[3],
                out_channels=hidden_channels[4],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            ),  # 2 15
            ConvBlock(
                bn=bn,
                activation=activation,
                in_channels=hidden_channels[4],
                out_channels=hidden_channels[5],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            )  # 1 7
        )

        self.avg_pool = nn.AvgPool2d((1, 7))


    def forward(self, scan):
        scan = self.conv_layers(scan)
        scan = self.avg_pool(scan).squeeze(-1).squeeze(-1)

        return scan



class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std):
        super().__init__()
        self.scan_dim = 480
        self.state_dim = 6

        self.scan_embeddinglayers = SensorEmbedding(bn=actor_bn, activation=actor_activation)

        # state
        self.fc1 = nn.Linear(self.state_dim, hidden_channels[-1]//2)
        self.bn1 = nn.BatchNorm1d(hidden_channels[-1] // 2)
        self.fc2 = nn.Linear(hidden_channels[-1]//2, hidden_channels[-1])
        self.bn2 = nn.BatchNorm1d(hidden_channels[-1])

        # scan & state
        self.fc3 = nn.Linear(hidden_channels[-1] * 2, hidden_channels[-1] * 2)
        self.bn3 = nn.BatchNorm1d(hidden_channels[-1] * 2)

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

        self.fc4 = nn.Linear(hidden_channels[-1] * 2, 2)
        self.tanh = nn.Tanh()

        self.act_mean = torch.Tensor(act_mean)
        self.act_std = torch.Tensor(act_std)

    def forward(self, obs):
        obs = obs.unsqueeze(1)
        scan, state = obs[:, :, :self.scan_dim*30], obs[:, :, self.scan_dim*30:].squeeze(1)   # 30 : stacked_size
        scan = scan.reshape(-1, 1, 30, 480)

        scan = self.scan_embeddinglayers(scan)

        state = F.relu(self.bn1(self.fc1(state)))
        state = F.relu(self.bn2(self.fc2(state)))

        out = torch.cat([scan, state], dim=-1)
        out = F.relu(self.bn3(self.fc3(out)))

        out = self.fc4(out)
        out = self.tanh(out)

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

        #### Conv2D ####
        self.scan_embeddinglayers = SensorEmbedding(bn=critic_bn, activation=critic_activation)

        # act & state
        self.fc1 = nn.Linear(self.state_dim + 2, hidden_channels[-1] // 2)
        self.fc2 = nn.Linear(hidden_channels[-1] // 2, hidden_channels[-1])

        # scan & state+act
        self.fc3 = nn.Linear(hidden_channels[-1] * 2, hidden_channels[-1] * 2)
        self.fc4 = nn.Linear(hidden_channels[-1] * 2, 1)

    def forward(self, obs, act):
        obs = obs.unsqueeze(1)
        scan, state = obs[:, :, :self.scan_dim*30], obs[:, :, self.scan_dim*30:].squeeze(1)
        scan = scan.reshape(-1, 1, 30, 480)
        scan = self.scan_embeddinglayers(scan)

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
