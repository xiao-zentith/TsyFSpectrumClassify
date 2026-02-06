import numpy as np
import matplotlib.pyplot as plt

# 角度列表
angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()

# 数据组
data_groups = [
    [3, 5, 2, 6],  # 第一组数据
    [1, 2, 4, 3],  # 第二组数据
    [0, 1, 3, 2],  # 第三组数据
    [2, 4, 3, 5]
]

# 设置标签
labels = ['A', 'B', 'C', 'D']
clist = ['blue', 'red', 'green', 'black', 'darkgreen', 'lime', 'gold', 'purple', 'green', 'cyan', 'salmon', 'grey',
         'mediumvioletred', 'darkkhaki', 'gray', 'darkcyan', 'violet', 'powderblue']

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# 为每组数据绘制雷达图
for i, data in enumerate(data_groups):
    ax.fill(angles, data, color=clist[i], alpha=0.25)
    ax.plot(angles, data, 'o', markeredgewidth=2, markeredgecolor=clist[i])
    ax.set_thetagrids(np.degrees(angles), labels)
    plt.grid(True)  # 添加网格线以更清晰地显示数据
    plt.pause(0.1)  # 暂停一段时间以便于观察每组数据的绘制过程

plt.show()