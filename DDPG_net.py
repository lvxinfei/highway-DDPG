import math
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import pprint
import highway_env
from torch.utils.tensorboard import SummaryWriter
use_cuda = torch.cuda.is_available()
# print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size ,init_w = 3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w,init_w)
        self.linear3.bias.data.uniform_(-init_w,init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w = 3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        # uniform_将tensor用从均匀分布中抽样得到的值填充。参数初始化
        self.linear3.weight.data.uniform_(-init_w, init_w)
        #也用用normal_(0, 0.1) 来初始化的，高斯分布中抽样填充，这两种都是比较有效的初始化方式
        self.linear3.bias.data.uniform_(-init_w, init_w)
        #其意义在于我们尽可能保持 每个神经元的输入和输出的方差一致。
        #使用 RELU（without BN） 激活函数时，最好选用 He 初始化方法，将参数初始化为服从高斯分布或者均匀分布的较小随机数
        #使用 BN 时，减少了网络对参数初始值尺度的依赖，此时使用较小的标准差(eg：0.01)进行初始化即可

        #但是注意DRL中不建议使用BN

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        # x = torch.sigmoid(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        #将经过tanh输出的值重新映射回环境的真实值内
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        #因为激活函数使用的是tanh，这里将环境输出的动作正则化到（-1，1）

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action


class DDPG(object):
    def __init__(self, action_dim, state_dim, hidden_dim, value_lr, policy_lr):
        super(DDPG,self).__init__()
        self.action_dim, self.state_dim, self.hidden_dim = action_dim, state_dim, hidden_dim
        self.batch_size = 32
        self.gamma = 0.99
        self.min_value = -np.inf
        self.max_value = np.inf
        self.soft_tau = 1e-2
        self.replay_buffer_size = 100000
        self.value_lr = value_lr
        self.policy_lr = policy_lr

        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)


        self.value_criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def ddpg_update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, self.min_value, self.max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

