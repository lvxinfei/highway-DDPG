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
device   = torch.device("cuda" if use_cuda else "cpu")

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
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(DDPG,self).__init__()
        self.action_dim, self.state_dim, self.hidden_dim = action_dim, state_dim, hidden_dim
        self.batch_size = 32
        self.gamma = 0.99
        self.min_value = -np.inf
        self.max_value = np.inf
        self.soft_tau = 2e-2
        self.replay_buffer_size = 7000
        self.value_lr = 5e-4
        self.policy_lr = 5e-4

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


env = gym.make("lvxinfei-v0")
env.configure(
{
    "observation": {
        "type": "OccupancyGrid",
        "features": ['presence', 'vx', 'vy', 'on_road'],
        "grid_size": [[-5, 5], [-7, 7]],
        "grid_step": [2, 2],#每个网格的大小
        "as_image": False,
        "align_to_vehicle_axes": True
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 500,
    "collision_reward": -100,
    "lane_centering_cost": 1,
    "action_reward": 0.3,
    "arrival_reward": 100,
    "controlled_vehicles": 1,
    "other_vehicles": 10,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
})


env.reset()
env = NormalizedActions(env)


print(env.observation_space.shape)
state_dim = env.observation_space.shape[2]*env.observation_space.shape[1]*env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print("状态维度"+str(state_dim))
print("动作维度"+str(action_dim))
# print(env.action_space)
hidden_dim = 256


ddpg = DDPG(action_dim, state_dim, hidden_dim)

max_steps = 350
rewards = []
batch_size = 32
VAR = 0.5  # control exploration


writer = SummaryWriter("./logs_train")
total_train_step = 0
#加载模型===========================================================
#加载上次训练好的模型，如果test_flag=True,则加载已保存的模型
test_flag=True#True  #True
if test_flag:
    ddpg = torch.load('./weights_test/ddpg_net.pth')
    # ddpg.value_net.load_state_dict(torch.load('./weights_test/ddpg_value_net.pth'))
    # ddpg.policy_net.load_state_dict(torch.load('./weights_test/ddpg_policy_net.pth'))
    # ddpg.target_value_net.load_state_dict(torch.load('./weights_test/ddpg_target_value_net.pth'))
    # ddpg.target_policy_net.load_state_dict(torch.load('./weights_test/ddpg_target_policy_net.pth'))
    
    print("模型加载成功！")



for step in range(max_steps):
    print("================第{}回合======================================".format(step+1))
    state = env.reset()
    state = torch.flatten(torch.tensor(state))
    episode_reward = 0
    done = False

    while not done:
        action = ddpg.policy_net.get_action(state)
        action[0] = np.clip(np.random.normal(action[0],VAR),-0.3,1) # 在动作选择上添加随机噪声
        action[1] = np.clip(np.random.normal(action[1],VAR),-1,1) # 在动作选择上添加随机噪声
        # action = np.clip(np.random.normal(action,VAR),-1,1)
        next_state, reward, done, info = env.step(action)
        next_state = torch.flatten(torch.tensor(next_state))
        ddpg.replay_buffer.push(state, action, reward, next_state, done)

        if len(ddpg.replay_buffer) > batch_size:
            VAR *= 0.9995    # decay the action randomness
            ddpg.ddpg_update()

        state = next_state
        episode_reward += reward
        env.render()

    total_train_step = total_train_step + 1
    if total_train_step % 10 == 0:
            writer.add_scalar("train_reward", episode_reward, total_train_step)
    rewards.append(episode_reward)
    print("回合奖励为：{}".format(episode_reward))
env.close()
writer.close()

#仅保存模型参数
# torch.save(ddpg, './weights_test/ddpg_net.pth')
# torch.save(ddpg.value_net.state_dict(), './weights_test/ddpg_value_net.pth')
# torch.save(ddpg.target_value_net.state_dict(), './weights_test/ddpg_target_value_net.pth')
# torch.save(ddpg.policy_net.state_dict(), './weights_test/ddpg_policy_net.pth')
# torch.save(ddpg.target_policy_net.state_dict(), './weights_test/ddpg_target_policy_net.pth')
# print("模型保存成功！")

plt.plot(rewards)
plt.savefig('./episode_reward.jpg')
