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
from highway_env.vehicle.behavior import IDMVehicle
import json
from collections import defaultdict



use_cuda = torch.cuda.is_available()
device  = torch.device("cuda" if use_cuda else "cpu")



env = gym.make("lvxinfei-v0")
env.reset()


ddpg = torch.load('./weights_test/ddpg_net1.pth')#直线形模型
ddpg1 = torch.load('./weights_test/ddpg_net2.pth')#曲线形模型

max_steps = 1
rewards = []
batch_size = 32
output1 = []
output2 = []
info_out = defaultdict(list)

with torch.no_grad():
    for step in range(max_steps):
        print("================第{}回合======================================".format(step+1))
        state = env.reset()
        state = torch.flatten(torch.tensor(state))
        done = False
        t=0
        while not done:
            if t>=10:
            	action = ddpg.policy_net.get_action(state)
            	print(0)
            elif t<10:
            	action = ddpg1.policy_net.get_action(state)
            	print(1)
            # action = ddpg.policy_net.get_action(state)
            next_state, reward, done, info = env.step(action)

            '''info字典中含有的车辆信息'''
            '''
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "vehicle heading": self.vehicle.heading,#车辆相对于大地坐标系的指向角，以pi为单位
            "action": action,
            'x': self.vehicle.position[0],
            'y': self.vehicle.position[1],
            "vx": self.vehicle.velocity[0],#速度与sin_h的乘积
            'vy': self.vehicle.velocity[1],
            'sin_h': self.vehicle.direction[1],
            "cos_h": self.vehicle.direction[0]
            '''
            # 对一些信息进行存储
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
            t=t+1
env.close()

with open("./JSON/v00.json", 'w', encoding='UTF-8') as f:
    f.write(json.dumps(info_out))

