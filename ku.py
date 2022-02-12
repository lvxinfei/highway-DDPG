# import gym
# import highway_env
# from matplotlib import pyplot as plt

# help(highway_env)#输出该文件夹的位置
# env = gym.make('lvxinfei-v0')
# env.reset()
# # env.render()
# plt.imshow(env.render(mode="rgb_array"))
# plt.show()


import gym
import highway_env
from matplotlib import pyplot as plt

env = gym.make('parking-ActionRepeat-v0')
env.config["longitudinal"] = True
obs = env.reset()
for i in range(0,1):
	obs, reward, done, info = env.step(env.action_space.sample())
	print(info)
	env.render()
plt.imshow(env.render(mode="rgb_array"))
plt.show()


