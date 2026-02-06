import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设你有一个二维矩阵形式的光谱数据
# 这里我们随机生成一些数据作为示例
np.random.seed(0)  # 为了结果的可重复性
spectral_data = np.random.rand(10, 10)  # 生成一个10x10的矩阵

# 创建X和Y坐标网格
x = np.arange(spectral_data.shape[1])
y = np.arange(spectral_data.shape[0])
X, Y = np.meshgrid(x, y)

# 创建一个图形和一个子图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制等高线图
ax.contour3D(X, Y, spectral_data, cmap='viridis')  # 使用viridis颜色图

# 添加颜色条
fig.colorbar(ax.contour3D(X, Y, spectral_data, 50, cmap='viridis'), ax=ax, shrink=0.5, aspect=5)

# 添加标题和坐标轴标签
ax.set_title('3D Contour Plot of Spectral Data')
ax.set_xlabel('Wavelength')
ax.set_ylabel('Intensity')
ax.set_zlabel('Value')

# 显示图形
plt.show()