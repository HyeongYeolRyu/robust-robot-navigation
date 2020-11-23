import torch
import torch.nn as nn
​
class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_h,
                 kernel_w,
                 stride_h,
                 stride_w):
        super(ConvBlock, self).__init__()
​
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            (kernel_h, kernel_w), (stride_h, stride_w)
        )
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(out_channels)
​
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)
​
        return x
​
class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        self.conv_block1 = ConvBlock(1, 32, 3, 40, 2, 4)
        self.maxpool2d = nn.MaxPool2d((1, 3), (1, 3))
        self.conv_block2 = ConvBlock(32, 64, 3, 3, 2, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4608, 600)
        self.fc2 = nn.Linear(6, 600)
        self.relu = nn.ReLU(True)
​
    def forward(self, obs, state):
        obs = self.conv_block1(obs)
        obs = self.maxpool2d(obs)
        obs = self.conv_block2(obs)
        obs = self.flatten(obs)
        obs = self.relu(self.fc1(obs))
        state = self.relu(self.fc2(state))
        out = torch.cat([obs, state], dim=1)
​
        return out
​
class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(1200, 512)
        # self.lstm = nn.LSTM()
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU(True)
​
    def forward(self, x):
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
​
        return out
​
class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(1202, 512)
        # self.lstm = nn.LSTM()
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU(True)
​
    def forward(self, embedding, action):
        x = torch.cat([embedding, action], dim=1)
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
​
        return out
​
if __name__ == '__main__':
    embedding_model = EmbeddingNetwork()
    actor_model = ActorNetwork()
    critic_model = CriticNetwork()
​
    batch_size = 1
    input_channel = 1
​
    obs = torch.randn(batch_size, input_channel, 30, 486)
    state = torch.randn(batch_size, 6)
​
    embedding = embedding_model(obs, state)
    action = actor_model(embedding)
    critic_output = critic_model(embedding, action)
    print(critic_output.size())
    print(action.size())