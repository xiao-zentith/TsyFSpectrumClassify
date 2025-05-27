import pandas as pd
import numpy as np
from matplotlib import rcParams
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 数据读取函数
def read_eem_from_excel(file_path):
    df = pd.read_excel(file_path, header=None)
    excitation_orig = df.iloc[0, 1:].values.astype(float)
    emission = df.iloc[1:, 0].values.astype(float)
    original_data = df.iloc[1:, 1:].values.T.astype(float)  # 注意转置
    return excitation_orig, emission, original_data


# 插值处理函数
def interpolate_excitation(original_data, excitation_orig, new_size=63):
    excitation_new = np.linspace(excitation_orig.min(), excitation_orig.max(), new_size)
    interpolated = np.zeros((new_size, original_data.shape[1]))

    for j in range(original_data.shape[1]):
        f = interp1d(excitation_orig, original_data[:, j],
                     kind='linear', fill_value="extrapolate")
        interpolated[:, j] = f(excitation_new)

    return excitation_new, interpolated


# 数据读取和插值
file_path = r"C:\Users\xiao\PycharmProjects\TsyFSpectrumClassify\dataset\dataset_preprocess\C6 + FITC\mixed_C1 + F9_extracted_with_C2 + F8_extracted.xlsx"
ex_orig, em_orig, data_orig = read_eem_from_excel(file_path)
ex_new, data_interp = interpolate_excitation(data_orig, ex_orig)

# 创建绘图网格
X_orig, Y_orig = np.meshgrid(em_orig, ex_orig)
X_interp, Y_interp = np.meshgrid(em_orig, ex_new)

# 1. 二维热图对比
plt.figure(figsize=(16, 6))

# 原始数据热图

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 或其他支持中文的字体，如 'Noto Sans CJK SC'
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.subplot(1, 2, 1)
plt.imshow(data_orig, aspect='auto',
           extent=[em_orig.min(), em_orig.max(), ex_orig.max(), ex_orig.min()],
           cmap='viridis')
plt.colorbar(label='荧光强度')
plt.title('原始EEM矩阵 (21×63)\n激发波长范围: {:.1f}-{:.1f} nm'.format(ex_orig.min(), ex_orig.max()))
plt.xlabel('发射波长 (nm)')
plt.ylabel('激发波长 (nm)')
plt.clim(vmin=data_orig.min(), vmax=data_orig.max())

# 插值后热图
plt.subplot(1, 2, 2)
plt.imshow(data_interp, aspect='auto',
           extent=[em_orig.min(), em_orig.max(), ex_new.max(), ex_new.min()],
           cmap='viridis')
plt.colorbar(label='荧光强度')
plt.title('插值后EEM矩阵 (63×63)\n激发波长间隔: {:.2f} nm'.format((ex_new.max() - ex_new.min()) / 62))
plt.xlabel('发射波长 (nm)')
plt.ylabel('激发波长 (nm)')
plt.clim(vmin=data_orig.min(), vmax=data_orig.max())

plt.tight_layout()

# 2. 三维曲面对比
fig = plt.figure(figsize=(16, 8))

# 原始数据曲面
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X_orig, Y_orig, data_orig, cmap='viridis',
                         linewidth=0, antialiased=False, alpha=0.7)
ax1.set_title('原始数据三维分布')
ax1.view_init(elev=30, azim=-45)
ax1.set_zlim(data_orig.min(), data_orig.max())

# 插值数据曲面
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X_interp, Y_interp, data_interp, cmap='plasma',
                         linewidth=0, antialiased=False, alpha=0.7)
ax2.set_title('插值后数据三维分布')
ax2.view_init(elev=30, azim=-45)
ax2.set_zlim(data_orig.min(), data_orig.max())

# 公共设置
for ax in [ax1, ax2]:
    ax.set_xlabel('发射波长 (nm)')
    ax.set_ylabel('激发波长 (nm)')
    ax.set_zlabel('强度')

plt.tight_layout()

# 3. 特征剖面对比
plt.figure(figsize=(16, 6))

# 激发方向特征对比
plt.subplot(1, 2, 1)
sample_em_idx = 30  # 选择中间发射波长
plt.plot(ex_orig, data_orig[:, sample_em_idx], 'bo-', label='原始数据')
plt.plot(ex_new, data_interp[:, sample_em_idx], 'r.-', label='插值数据')
plt.title(f'发射波长 {em_orig[sample_em_idx]:.1f} nm 处激发光谱对比')
plt.xlabel('激发波长 (nm)')
plt.ylabel('强度')
plt.legend()

# 发射方向特征对比
plt.subplot(1, 2, 2)
sample_ex_idx = 10  # 选择中间激发波长
plt.plot(em_orig, data_orig[sample_ex_idx, :], 'go-', label='原始数据')
plt.plot(em_orig, data_interp[sample_ex_idx * 3, :], 'm.-', label='插值对应点')  # 近似对应位置
plt.title(f'激发波长 {ex_new[sample_ex_idx * 3]:.1f} nm 处发射光谱对比')
plt.xlabel('发射波长 (nm)')
plt.ylabel('强度')
plt.legend()

plt.tight_layout()
plt.show()