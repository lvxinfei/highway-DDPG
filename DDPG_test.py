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
import json
from collections import defaultdict


use_cuda = torch.cuda.is_available()
device  = torch.device("cuda" if use_cuda else "cpu")



env = gym.make("lvxinfei-v1")
env.reset()


ddpg = torch.load('./weights_test/ddpg_net1-1.pth')


max_steps = 1
rewards = []
batch_size = 32
speed = []
info_out = defaultdict(list)


with torch.no_grad():
    for step in range(max_steps):
        print("================第{}回合======================================".format(step+1))
        state = env.reset()
        state = torch.flatten(torch.tensor(state))
        done = False

        while not done:
            action = ddpg.policy_net.get_action(state)
            next_state, reward, done, info = env.step(action)
            #对一些信息进行存储
            info_out["speed"].append(info['speed'])
            info_out["x"].append(info['x'])
            info_out["y"].append(info['y'])
            info_out["vx"].append(info['vx'])
            info_out["vy"].append(info['vy'])
            info_out["sin_h"].append(info['sin_h'])
            info_out["cos_h"].append(info['cos_h'])
            info_out["vehicle heading"].append(info['vehicle heading'])
            info_out['road heading'].append(info['road heading'])

            # print(info)
            next_state = torch.flatten(torch.tensor(next_state))
            state = next_state
            env.render()
env.close()

with open("./JSON/v1.json", 'w', encoding='UTF-8') as f:
    f.write(json.dumps(info_out))
