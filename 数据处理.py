import json
import matplotlib.pyplot as plt
import math
import numpy as np
#读取JSON格式文件的内容并转换为字典
with open("./JSON/v1.json", 'r', encoding = 'UTF-8') as f:
    d3 = json.load(f)
print("\n读取JSON格式文件的内容并转换为字典：\n")
# print(d3)

#对道路的heading信息进行转换，防止出现角度跳跃现象
sita = d3["road heading"]
for i in range(1, len(sita)):
    if sita[i] - sita[i - 1] > 1.5 * math.pi:
        sita[i] = sita[i] - 2 * math.pi
    if sita[i] - sita[i - 1] < -1.5 * math.pi:
        sita[i] = sita[i] + 2 * math.pi

#输出车辆以及道路的朝向角度信息
plt.scatter(range(len(d3["vehicle heading"])),d3["vehicle heading"],color="y")
plt.scatter(range(len(sita)),sita)
# plt.savefig('./episode_reward.jpg')#保存输出的图片
plt.show()

#输出车辆与道路中线的夹角
j1 = np.array(sita)-np.array(d3["vehicle heading"])
plt.scatter(range(len(sita)),j1)
plt.plot(j1)
plt.show()

#输出侧向速度
plt.scatter(range(len(sita)),d3["speed"]*np.sin(j1))
plt.plot(range(len(sita)),d3["speed"]*np.sin(j1))
plt.show()
