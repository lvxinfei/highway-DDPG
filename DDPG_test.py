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
from DDPG_net import * 

use_cuda = torch.cuda.is_available()
device  = torch.device("cuda" if use_cuda else "cpu")



env = gym.make("lvxinfei-v2")
env.reset()


ddpg = torch.load('./weights_test/ddpg_net0.pth')


max_steps = 10
rewards = []
batch_size = 32
speed = []
with torch.no_grad():
    for step in range(max_steps):
        print("================第{}回合======================================".format(step+1))
        state = env.reset()
        state = torch.flatten(torch.tensor(state))
        done = False

        while not done:
            action = ddpg.policy_net.get_action(state)
            next_state, reward, done, info = env.step(action)
            speed.append(info['speed'])
            print(info)
            next_state = torch.flatten(torch.tensor(next_state))
            state = next_state
            env.render()
env.close()

# plt.plot(speed)
# plt.show()
