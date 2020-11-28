import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


""" sinusoid position embedding """
def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table

    # print (pos_encoding.shape)
    # plt.pcolormesh(pos_encoding, cmap='RdBu')
    # plt.xlabel('Depth')
    # plt.xlim((0, d_hidn))
    # plt.ylabel('Position')
    # plt.colorbar()
    # plt.show()

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_h,
                 kernel_w,
                 stride_h,
                 stride_w):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            (kernel_h, kernel_w), (stride_h, stride_w), bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x) # TODO: check
        x = self.relu(x)

        return x


class Nav2dEmbeddingNetwork(nn.Module):
    def __init__(self, sensor_dim=480, n_stack=30, n_channel=1, partition=False, state_dim=6):
        super(Nav2dEmbeddingNetwork, self).__init__()
        self.sensor_dim = sensor_dim
        self.n_stack = n_stack
        self.n_channel = n_channel
        self.partition = partition
        self.state_dim = state_dim

        if self.partition == False:
            self.conv_block1 = ConvBlock(1, 32, 3, 40, 2, 4)
            self.maxpool2d = nn.MaxPool2d((1, 3), (1, 3))
            self.conv_block2 = ConvBlock(32, 64, 3, 3, 2, 3)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(4608, 600)
        else:
            self.n_partition = 8
            self.part_dim = (self.sensor_dim//self.n_partition)
            n_seq = self.n_stack
            d_hidn = self.sensor_dim
            self.pos_encoding = torch.Tensor(get_sinusoid_encoding_table(n_seq, d_hidn)).unsqueeze(0).unsqueeze(0).repeat(5,1,1,1)
            self.conv_block1 = ConvBlock(2, 32, 3, 5, 2, 2)
            self.conv_block2 = ConvBlock(32, 64, 3, 5, 2, 2)
            self.conv_block3 = ConvBlock(64, 64, 3, 5, 2, 2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(4096, 600)

        self.fc2 = nn.Linear(6, 600)
        self.relu = nn.ReLU(True)

    def forward(self, obs): # (in) obs: Tensor (dim:[-1,14406])
        if self.partition == True:
            if obs.is_cuda:
                self.pos_encoding = self.pos_encoding.cuda()
            else:
                self.pos_encoding = self.pos_encoding.cpu()

        sensor_obs = obs[:,:-6].reshape(-1,self.n_channel,self.n_stack,self.sensor_dim)    # Bx1x30x480
        robot_state = obs[:,-6:]                                                           # Bx6

        if self.partition == False:                                                        #-----No partition-----
            sensor_obs = self.conv_block1(sensor_obs)                                          # Bx32x14x111
            sensor_obs = self.maxpool2d(sensor_obs)                                            # Bx32x14x37
            sensor_obs = self.conv_block2(sensor_obs)                                          # Bx64x6x12
            sensor_obs = self.flatten(sensor_obs)                                              # Bx4608
            sensor_obs = self.relu(self.fc1(sensor_obs))                                       # Bx600
        else:                                                                              #----- partition ------
            sensor_obs = torch.cat([sensor_obs,self.pos_encoding], dim=1)                      # Bx2x30x480
            split_set = []
            for i in range(self.n_partition):
                split_set.append(sensor_obs[:,:,:,self.part_dim*i:self.part_dim*(i+1)])        # Bx2x30x60
            for i in range(self.n_partition):
                split_set[i] = self.conv_block1(split_set[i])                                  # Bx32x14x28
                split_set[i] = self.conv_block2(split_set[i])                                  # Bx64x6x12
                split_set[i] = self.conv_block3(split_set[i])                                  # Bx64x2x4
                split_set[i] = self.flatten(split_set[i])                                      # Bx512
            # TODO: apply attention
            sensor_obs = torch.cat([split_set[0],split_set[1],split_set[2],split_set[3],split_set[4],split_set[5],split_set[6],split_set[7]],dim=1) # Bx4096
            sensor_obs = self.relu(self.fc1(sensor_obs))                                       # Bx600

        robot_state = self.relu(self.fc2(robot_state))                                     # Bx600

        out = torch.cat([sensor_obs, robot_state], dim=1)
        return out                                                                         # Bx1200


class Nav2dActorNetwork(nn.Module):
    def __init__(self, act_mean, act_std, partition=False):
        super(Nav2dActorNetwork, self).__init__()
        self.act_mean = torch.Tensor(act_mean)
        self.act_std = torch.Tensor(act_std)

        self.embedding_network = Nav2dEmbeddingNetwork(partition=partition)
        self.fc1 = nn.Linear(1200, 512)
        self.bn1 = nn.BatchNorm1d(512)
        # self.lstm = nn.LSTM()
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.embedding_network(x)
        x = self.relu(self.bn1(self.fc1(x)))
        out = self.tanh(self.fc2(x))
        if out.is_cuda:
            return (self.act_std.cuda() * out) + self.act_mean.cuda()
        else:
            return (self.act_std * out) + self.act_mean


class Nav2dCriticNetwork(nn.Module):
    def __init__(self):
        super(Nav2dCriticNetwork, self).__init__()

        self.embedding_network = Nav2dEmbeddingNetwork()
        self.fc1 = nn.Linear(1202, 512)
        # self.lstm = nn.LSTM()
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU(True)

    def forward(self, obs, action):
        obs = self.embedding_network(obs)
        x = torch.cat([obs, action], dim=1)
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out

class Nav2dActorCritic(nn.Module):
    def __init__(self,observation_space, action_space, partition=False):
        super(Nav2dActorCritic, self).__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        act_std = (action_space.high - action_space.low)/2
        act_mean = (action_space.high + action_space.low)/2

        self.pi = Nav2dActorNetwork(act_mean, act_std, partition=partition)
        self.q = Nav2dCriticNetwork()

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


if __name__ == '__main__':
    print("===============2d AC=================")
    ac = Nav2dActorCritic()
    print(ac)
    batch_size = 5
    input_channel = 1
    sensor_obs = torch.randn(batch_size, input_channel, 30, 480)
    sensor_obs = sensor_obs.reshape(-1,14400)
    robot_state = torch.randn(batch_size, 6)
    obs = torch.cat((sensor_obs,robot_state),1)
    print(obs.size())
    out = ac.act(obs)
    print(out)
    print(out.shape)
    print("===============CUDA Test=================")
    obs = torch.cat((sensor_obs,robot_state),1).cuda()
    print(obs.size())
    ac = ac.cuda()
    out = ac.pi(obs)
    print(out)
    print(out.shape)
    print("===============partition=================")
    ac = Nav2dActorCritic(partition=True)
    print(ac)
    batch_size = 5
    input_channel = 1
    sensor_obs = torch.randn(batch_size, input_channel, 30, 480)
    sensor_obs = sensor_obs.reshape(-1,14400)
    robot_state = torch.randn(batch_size, 6)
    obs = torch.cat((sensor_obs,robot_state),1)
    print(obs.size())
    out = ac.act(obs)
    print(out)
    print(out.shape)
    print("===============CUDA Test=================")
    obs = torch.cat((sensor_obs,robot_state),1).cuda()
    print(obs.size())
    ac = ac.cuda()
    out = ac.pi(obs)
    print(out)
    print(out.shape)