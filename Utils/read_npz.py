import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from torch.ao.ns.fx.utils import compute_cosine_similarity

from Utils.compute_pearson import pearson_corr_matrix
from Utils.compute_relative_error import calculate_relative_error


def read_npz_file(file_path):
    # 加载.npz文件
    data = np.load(file_path)

    # 打印所有数组的名称
    print("Arrays in the .npz file:")
    for key in data.files:
        print(f"- {key}")

    # 检查所需的数组是否存在
    required_keys = ['input', 'target1', 'target2', 'pred1', 'pred2']
    missing_keys = [key for key in required_keys if key not in data.files]

    if missing_keys:
        print(f"\nThe file does not contain the following arrays: {missing_keys}. Please check the available arrays listed above.")
        return None, None, None, None, None

    # 提取所需的数据
    input_data = data['input']
    target1_data = data['target1']
    target2_data = data['target2']
    pred1_data = data['pred1']
    pred2_data = data['pred2']
    plot_spectra_contour(input_data, target1_data, target2_data, pred1_data, pred2_data)

    return input_data, target1_data, target2_data, pred1_data, pred2_data

def plot_spectra_contour(input_data, target1_data, target2_data, pred1_data, pred2_data):
    # 定义横坐标和纵坐标的范围
    emission_wavelengths = np.arange(310, 621, 5)  # 激发光谱：300-500nm，间隔为5nm
    excitation_wavelengths = np.arange(300, 501, 10)   # 发射光谱：310-620nm，间隔为5nm

    # 创建网格
    X, Y = np.meshgrid(excitation_wavelengths, emission_wavelengths)

    # 绘制等高线图
    fig = plt.figure(figsize=(18, 8))

    # 自定义colormap：紫、靛、蓝、绿、黄、橙、赤
    cmap = 'jet'

    # Input 等高线图
    ax_input1 = fig.add_subplot(2, 3, 1)
    contour_input1 = ax_input1.contourf(X, Y, input_data, cmap=cmap, levels=50)
    ax_input1.set_title('Input Spectra Contour')
    ax_input1.set_xlabel('Excitation Wavelength /nm')
    ax_input1.set_ylabel('Emission Wavelength /nm')
    fig.colorbar(contour_input1, ax=ax_input1, orientation='vertical')

    ax_input2 = fig.add_subplot(2, 3, 4)
    contour_input2 = ax_input2.contourf(X, Y, input_data, cmap=cmap, levels=50)
    ax_input2.set_title('Input Spectra Contour')
    ax_input2.set_xlabel('Excitation Wavelength /nm')
    ax_input2.set_ylabel('Emission Wavelength /nm')
    fig.colorbar(contour_input2, ax=ax_input2, orientation='vertical')

    # Target1 等高线图
    ax1 = fig.add_subplot(2, 3, 2)
    contour1 = ax1.contourf(X, Y, target1_data, cmap=cmap, levels=500)
    ax1.set_title('PARAFAC 1 Spectra Contour')
    ax1.set_xlabel('Excitation Wavelength /nm')
    ax1.set_ylabel('Emission Wavelength /nm')
    fig.colorbar(contour1, ax=ax1, orientation='vertical')

    # Target2 等高线图
    ax2 = fig.add_subplot(2, 3, 5)
    contour2 = ax2.contourf(X, Y, target2_data, cmap=cmap, levels=500)
    ax2.set_title('PARAFAC 2 Spectra Contour')
    ax2.set_xlabel('Excitation Wavelength /nm')
    ax2.set_ylabel('Emission Wavelength /nm')
    fig.colorbar(contour2, ax=ax2, orientation='vertical')

    # Pred1 等高线图
    ax3 = fig.add_subplot(2, 3, 3)
    contour3 = ax3.contourf(X, Y, pred1_data, cmap=cmap, levels=500)
    ax3.set_title('UNet 1 Spectra Contour')
    ax3.set_xlabel('Excitation Wavelength /nm')
    ax3.set_ylabel('Emission Wavelength /nm')
    fig.colorbar(contour3, ax=ax3, orientation='vertical')

    # Pred2 等高线图
    ax4 = fig.add_subplot(2, 3, 6)
    contour4 = ax4.contourf(X, Y, pred2_data, cmap=cmap, levels=500)
    ax4.set_title('UNet 2 Spectra Contour')
    ax4.set_xlabel('Excitation Wavelength /nm')
    ax4.set_ylabel('Emission Wavelength /nm')
    fig.colorbar(contour4, ax=ax4, orientation='vertical')

    plt.tight_layout()
    plt.show()

    print(np.nanmean(pearson_corr_matrix(target2_data, pred2_data)))
    print(np.nanstd(pearson_corr_matrix(target2_data, pred2_data)))
    print(calculate_relative_error(target1_data, pred1_data))
    print(calculate_relative_error(target1_data, pred1_data))


# 示例用法
file_path = r'..\regression_dataset\dataset_result\C6 + HPTS\fold_0\test_sample_1_data.npz'  # 替换为你的.npz文件路径
read_npz_file(file_path)



