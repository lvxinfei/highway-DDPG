import json
import matplotlib.pyplot as plt
import math
import numpy as np
#读取JSON格式文件的内容并转换为字典—多模型控制
with open("./JSON/v00.json", 'r', encoding = 'UTF-8') as f:
    d_m = json.load(f)
print("\n读取JSON格式文件的内容并转换为字典：\n")
# print(d3)
#读取JSON格式文件的内容并转换为字典-单一模型控制
with open("./JSON/v0.json", 'r', encoding = 'UTF-8') as f:
    d_1 = json.load(f)
print("\n读取JSON格式文件的内容并转换为字典：\n")



#对道路的heading信息进行转换，防止出现角度跳跃现象
for i in range(1, len(d_m["road heading"])):
    if d_m["road heading"][i] - d_m["road heading"][i - 1] > 1.5 * math.pi:
        d_m["road heading"][i] = d_m["road heading"][i] - 2 * math.pi
    if d_m["road heading"][i] - d_m["road heading"][i - 1] < -1.5 * math.pi:
        d_m["road heading"][i] = d_m["road heading"][i] + 2 * math.pi

for i in range(1, len(d_1["road heading"])):
    if d_1["road heading"][i] - d_1["road heading"][i - 1] > 1.5 * math.pi:
        d_1["road heading"][i] = d_1["road heading"][i] - 2 * math.pi
    if d_1["road heading"][i] - d_1["road heading"][i - 1] < -1.5 * math.pi:
        d_1["road heading"][i] = d_1["road heading"][i] + 2 * math.pi


#输出车辆与道路中线的夹角
j_1 = np.array(d_1["road heading"])-np.array(d_1["vehicle heading"])
plt.scatter(range(len(j_1)),j_1)
plt.plot(j_1,color="b")

j_m = np.array(d_m["road heading"])-np.array(d_m["vehicle heading"])
plt.scatter(range(len(j_m)),j_m)
plt.plot(j_m,color="r")

plt.show()

#输出侧向速度
plt.scatter(range(len(d_1["road heading"])),d_1["speed"]*np.sin(j_1))
plt.plot(range(len(d_1["road heading"])),d_1["speed"]*np.sin(j_1),color="b")

plt.scatter(range(len(d_m["road heading"])),d_m["speed"]*np.sin(j_m))
plt.plot(range(len(d_m["road heading"])),d_m["speed"]*np.sin(j_m),color="r")

plt.show()

#两个控制方式速度的对比
plt.plot(d_1["speed"], color="b")
plt.plot(d_m["speed"],color="r")
plt.show()