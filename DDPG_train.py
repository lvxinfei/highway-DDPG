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
import DDPG_net#DDPG网络结构

use_cuda = torch.cuda.is_available()
# print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")


env = gym.make("lvxinfei-v0")
env.reset()
# env = NormalizedActions(env)


print(env.observation_space.shape)
state_dim = env.observation_space.shape[2]*env.observation_space.shape[1]*env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print("状态维度"+str(state_dim))
print("动作维度"+str(action_dim))
# print(env.action_space)

hidden_dim = 256
value_lr = 1e-3
policy_lr = 1e-4

# ddpg = DDPG(action_dim, state_dim, hidden_dim, value_lr, policy_lr)
ddpg = DDPG_net.DDPG(action_dim, state_dim, hidden_dim, value_lr, policy_lr)

max_steps = 1000
rewards = []
batch_size = 32
VAR = 1  # control exploration


writer = SummaryWriter("./logs_train")
total_train_step = 0
#加载模型===========================================================
#加载上次训练好的模型，如果test_flag=True,则加载已保存的模型
test_flag=False#True  #False
if test_flag:
    ddpg = torch.load('./weights_test/ddpg_net1.pth')
    # ddpg.value_net.load_state_dict(torch.load('./weights_test/ddpg_value_net.pth'))
    # ddpg.policy_net.load_state_dict(torch.load('./weights_test/ddpg_policy_net.pth'))
    # ddpg.target_value_net.load_state_dict(torch.load('./weights_test/ddpg_target_value_net.pth'))
    # ddpg.target_policy_net.load_state_dict(torch.load('./weights_test/ddpg_target_policy_net.pth'))
    
    print("模型加载成功！")


scheduler = torch.optim.lr_scheduler.ExponentialLR(ddpg.value_optimizer, gamma=0.99)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(ddpg.policy_optimizer, gamma=0.99)

for step in range(max_steps):
    print("================第{}回合======================================".format(step+1))
    state = env.reset()
    state = torch.flatten(torch.tensor(state))
    episode_reward = 0
    done = False
    number = 0 #用于计算车辆在一个回合走了多少步，以便求平均回报

    while not done:
        action = ddpg.policy_net.get_action(state)
        action[0] = np.clip(np.random.normal(action[0],VAR),-1,1) # 在动作选择上添加随机噪声
        action[1] = np.clip(np.random.normal(action[1],VAR),-1,1) # 在动作选择上添加随机噪声
        # print(action)
        # action = np.clip(np.random.normal(action,VAR),-1,1)
        next_state, reward, done, info = env.step(action)
        next_state = torch.flatten(torch.tensor(next_state))
        ddpg.replay_buffer.push(state, action, reward, next_state, done)

        if len(ddpg.replay_buffer) > batch_size:
            VAR *= 0.9995    # decay the action randomness
            ddpg.ddpg_update()

        state = next_state
        episode_reward += reward
        number = number + 1
        env.render()

    total_train_step = total_train_step + 1
    if total_train_step % 5 == 0:
            writer.add_scalar("train_reward", episode_reward/number, total_train_step)
    rewards.append(episode_reward/number)

    print("回合平均累积回报为：{} | 价值网络学习率为：{} | 策略网络学习率为：{}".format(episode_reward/number,
    	ddpg.value_optimizer.state_dict()['param_groups'][0]['lr'],
    	ddpg.policy_optimizer.state_dict()['param_groups'][0]['lr']))
    if step != 0 and step % 50 == 0:#每10回合，学习率衰减1次
    	scheduler.step()
    	scheduler2.step()
env.close()
writer.close()

#仅保存模型参数
torch.save(ddpg, './weights_test/ddpg_net0-1.pth')
# torch.save(ddpg.value_net.state_dict(), './weights_test/ddpg_value_net.pth')
# torch.save(ddpg.target_value_net.state_dict(), './weights_test/ddpg_target_value_net.pth')
# torch.save(ddpg.policy_net.state_dict(), './weights_test/ddpg_policy_net.pth')
# torch.save(ddpg.target_policy_net.state_dict(), './weights_test/ddpg_target_policy_net.pth')
print("模型保存成功！")

plt.plot(rewards)
plt.show()
# plt.savefig('./episode_reward.jpg')
