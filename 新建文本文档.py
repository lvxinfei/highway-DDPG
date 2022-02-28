import json
import matplotlib.pyplot as plt


#读取JSON格式文件的内容并转换为字典
with open("1.json", 'r', encoding = 'UTF-8') as f:
    d3 = json.load(f)
print("\n读取JSON格式文件的内容并转换为字典：\n")
# print(d3)
plt.scatter(range(len(d3["vehicle heading"])),d3["vehicle heading"],color="y")
plt.scatter(range(len(d3["road heading"])),d3["road heading"])
plt.show()