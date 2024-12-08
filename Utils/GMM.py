import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设我们有一个包含三维光谱数据的numpy数组
# 这里我们创建一些假数据作为示例
np.random.seed(42)  # 设置随机种子以保证结果可复现
num_samples = 500  # 原始数据点数量
original_data = np.vstack([
    np.random.multivariate_normal([1, 2, 3], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], num_samples),
    np.random.multivariate_normal([5, 6, 7], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], num_samples)
])

# 训练GMM
n_components = 2  # 高斯成分的数量可以根据实际情况调整
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(original_data)

# 生成新数据
num_generated_samples = 1000  # 要生成的新样本数量
generated_data, generated_labels = gmm.sample(num_generated_samples)

# 可视化原始和生成的数据
fig = plt.figure(figsize=(12, 6))

# 绘制原始数据
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2], c='blue', marker='o', label='Original Data')
ax1.set_title('Original Spectral Data')
ax1.legend()

# 绘制生成的数据
ax2 = fig.add_subplot(122, projection='3d')
scatter = ax2.scatter(generated_data[:, 0], generated_data[:, 1], generated_data[:, 2], c=generated_labels, marker='x', label='Generated Data')
ax2.set_title('Generated Spectral Data')
ax2.legend()

plt.show()

# 打印生成的数据样本
print("Generated spectral data samples:")
print(generated_data[:5])  # 打印前五个生成的数据样本