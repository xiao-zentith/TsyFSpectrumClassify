import os
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
from scipy.signal import savgol_filter  # 需要安装scipy


def read_spectrum_data(file_path):
    x = []
    y = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) == 2:
                x.append(float(data[0]))
                y.append(float(data[1]))
    return np.array(x), np.array(y)


def plot_spectra(folder_path, window_length=11, polyorder=3):
    # 设置全局字体为Arial
    plt.rcParams['font.family'] = 'Arial'

    # 创建自定义颜色循环
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    plt.rc('axes', prop_cycle=cycler(color=colors))

    # 创建figure并设置尺寸
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111)

    # 读取并绘制数据
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            x, y = read_spectrum_data(file_path)

            # 数据平滑处理（Savitzky-Golay滤波器）
            if len(y) > window_length:
                y_smooth = savgol_filter(y, window_length, polyorder)
            else:
                y_smooth = y  # 数据点太少时不进行平滑

            # 绘制平滑后的曲线
            plt.plot(x, y_smooth,
                     linewidth=2.5,
                     alpha=0.9,
                     label=filename.split('.')[0])

    # 设置坐标轴参数
    ax.tick_params(axis='both', which='major',
                   direction='in', length=6,
                   width=1.5, labelsize=16)

    # 设置边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 隐藏上、右边框
    ax.spines[['right', 'top']].set_visible(False)

    # 设置标签和标题
    plt.xlabel('Wavelength (nm)', fontsize=20, labelpad=10)
    plt.ylabel('Intensity ', fontsize=20, labelpad=10)
    plt.title('Emission Spectra', fontsize=22, pad=20)

    # 设置图例
    legend = plt.legend(fontsize=12, frameon=True,
                        loc='best', fancybox=False,
                        edgecolor='black',
                        facecolor='white')
    legend.get_frame().set_linewidth(1.0)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.3)

    # 设置坐标轴范围
    plt.xlim(left=min(x), right=max(x))

    # 调整布局
    plt.tight_layout()

    # 保存高分辨率图片（可选）
    # plt.savefig('emission_spectra.png', dpi=300, bbox_inches='tight')

    plt.show()


# 使用示例：
folder_path = r'C:\Users\xiao\Desktop\academic_papers\data\em_spectrum'
plot_spectra(folder_path)